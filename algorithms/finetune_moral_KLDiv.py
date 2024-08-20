import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from dataclasses import dataclass
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

import time
from ppo import Args, Agent, make_env
from llm_moral import call_llm_with_state_action,create_llm_env,few_shot_prompt_training
from dempster_shafer import belief_to_reward

NUM_MORAL = 5

credences = np.zeros((5, NUM_MORAL))
# Set the diagonal elements
for i in range(NUM_MORAL):
    credences[i, i] = 1
    
# model_name = "llama3"
model_name = "gpt-4o-mini"
api_key = os.environ.get("OPENAI_API_KEY", "none")
api_key_coss = os.environ.get("OPENAI_API_KEY_COSS", "none")
model = create_llm_env(api_key_coss,model_name)
final_prompt = few_shot_prompt_training()

def log(logger,writer,question_response_dict,step,global_step,reward_dict,action,frame=None):   
    if args.write_to_csv==False:
        text = f"Step {step}\n"
        i = 0
        for key, value in question_response_dict.items():
            text += f"-------Question Prompt with credence index - {i}-------\n {key}\n -------Response Prompt-------\n{value}\n--------------------------------------\n"
            i+=1
            
        writer.add_text("LLM Prompts", f"\n{frame if frame is not None else ''}" + text, global_step)
    else:
        for key, value in question_response_dict.items():
            logger.log(step=step, question=key, response=value, reward=reward_dict, action=action)

def kl_div(p1, p2):
    total = 0.
    for idx in range(len(p1)):
        total += -p1[idx]*np.log(p2[idx]/p1[idx])
    return total
    
## OVERRIDES
@dataclass
class FineTuneArgs(Args):
    num_steps: int = 128 # note it is 64 for Milk
    total_timesteps: int = 2000*num_steps
    num_envs: int = 1
    update_epochs: int = 16
    anneal_lr: bool = False
    load_model: str = "runs/Driving__ppo__1__1723727577/ppo_base.cleanrl_model"
    write_to_csv: bool = True

if __name__ == "__main__":
    import pickle
    args = tyro.cli(FineTuneArgs)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    env_id = args.env_id.split(':')[-1] if ':' in args.env_id else args.env_id
    run_name = f"{env_id}__{args.exp_name}__{args.seed}__moral"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    logger = Logger(f"runs/{run_name}/log.csv")
    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.load_model))
    agent.critic = agent.reset_critic(envs) # why? 
    #This is the reference model (frozen) fo KL divergence
    agent_ref = Agent(envs).to(device)
    agent_ref.load_state_dict(torch.load(args.load_model)) 


    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_ref = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    total_token_usage = 0
    history_path = f'runs/{run_name}/{model_name}_llm_cache.pickle'
    history = {}
    if os.path.isfile(history_path):
        with open(history_path, 'rb') as handle:
            history = pickle.load(handle)
    for iteration in range(0, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                action_ref, logprob_ref, _, value_ref = agent_ref.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            logprobs_ref[step] = logprob_ref
            
            kl = kl_div(logprob_ref,logprob)
            kl_penalty_factor = 2 # based on Moral paper https://github.com/kristery/EthicsShaping/blob/master/Drive/hsarsa_n.py
            non_score_reward = (kl_penalty_factor * kl).numpy()
            # print(non_score_reward)
            the_actions = action.cpu().numpy()
            # TRY NOT TO MODIFY: execute the game and log data.
            shaping_reward = []
            for i in range(args.num_envs):
                unwrapped_env = envs.envs[i].unwrapped
                envstate = envs.observations[i] # the unwrapped env might not have a flat observation space
                if tuple(envstate) not in history:
                    state_text, action_text = unwrapped_env.state_as_text()
                    actionsets = [frozenset([str(k)]) for k in unwrapped_env.action_mapper.keys()] #TODO: review str casting 
                    scenario_prompt = unwrapped_env.get_scenario_prompt()
                    beliefs, question_response_dict,token_usage = call_llm_with_state_action(scenario_prompt,actionsets,state_text,action_text,credences,model,final_prompt) 
                    total_token_usage+=token_usage            
                    reward_dict = belief_to_reward(beliefs, actionsets)
                    history[tuple(envstate)] = reward_dict
                    if step%10==0: #log after every 10 steps - TODO: Make logging step as variable
                        log(logger,writer,question_response_dict,step,global_step,reward_dict,action.cpu().numpy(), frame=unwrapped_env.render())
                else:
                    print("Note: using cached LLM response")
                    reward_dict = history[tuple(envstate)]
                shaping_reward.append(reward_dict[frozenset([str(the_actions[i])])])
                
                writer.add_text("Reward & Action", f"Step {step}\n{reward_dict}\n {action}\n", global_step=global_step)
                # Cache state-action prompts to save processing time
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            print(f"total token usage at step {global_step} = {total_token_usage}")
            # The shaping reward is 1-p_sensor, to strongly disincentivise taking the least moral action.
            # shaping_reward = np.add(shaping_reward, -1)
            # reward = np.add(reward, shaping_reward)
            reward = shaping_reward + non_score_reward 
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar(f"charts/episodic_{info['metric1'][0]}", info["metric1"][1], global_step)
                        writer.add_scalar(f"charts/episodic_{info['metric2'][0]}", info["metric2"][1], global_step)

        #Adding logs to tensorboard for LLM question text and response text
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and (iteration%5==0 or iteration==args.num_iterations):
            model_path = f"runs/{run_name}/kl_div/{args.exp_name}_{iteration}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

            with open(history_path, 'wb') as handle:
                pickle.dump(history, handle)
    envs.close()
    writer.close()
