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
from torch.distributions.categorical import Categorical

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
api_key = os.environ.get("OPENAI_API_KEY_COSS", "none")
model = create_llm_env(api_key,model_name)
final_prompt = few_shot_prompt_training()
agent_pos_update_t = [(4,5),(5,6)]
agent_pos_update = [(2,3)]
env_State_temp = [9.0, 6.0, 7.0, 7.0, 6.0, 7.0, 5.0, 5.0]
    
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


kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)

def kl_div(p,q): 
    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    # Avoid division by zero and log(0) by adding a small value (epsilon)
    epsilon = 1e-10
    p = p+epsilon
    q = q+epsilon
    # divergence = np.sum(p*np.log(p/q))
    divergence = (np.exp(p)* (p - q)).sum()
    return divergence
    
## OVERRIDES
@dataclass
class FineTuneArgs(Args):
    num_steps: int = 64 # note it is 64 for Milk
    total_timesteps: int = 10000*num_steps
    num_envs: int = 1
    update_epochs: int = 16
    anneal_lr: bool = False
    # load_model: str = "runs/Driving__ppo__1__1724832763/ppo.cleanrl_model"
    load_model: str = "runs/FindMilk-v4__ppo__1__1724503897/ppo.cleanrl_model" #The Milk base model that gave us good result. KL factor of 2
    load_model_ref: str = "runs/FindMilk-v4__ppo__1__1724503897/ppo.cleanrl_model"
    load_from: int = 0
    write_to_csv: bool = True
kwargs = {'validate': True,
          'heuristic': False}

if __name__ == "__main__":
    import pickle
    args = tyro.cli(FineTuneArgs)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    env_id = args.env_id.split(':')[-1] if ':' in args.env_id else args.env_id
    if args.load_model_ref is None:
        args.load_model_ref = args.load_model
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
    writer = SummaryWriter(f"runs/{run_name}/kl_div/", filename_suffix=model_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    logger = Logger(f"runs/{run_name}/{model_name}_log.csv")
    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, **kwargs) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.load_model))
    agent.critic = agent.reset_critic(envs) # why? 
    #This is the reference model (frozen) fo KL divergence
    agent_ref = Agent(envs).to(device)
    agent_ref.load_state_dict(torch.load(args.load_model_ref)) 


    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
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
            
    #Load Human Policy 
    hpolicy = {}
    actions = range(4)
    with open('runs/human_policy/hpolicy_milk.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    for key in trajectory:
        if key[0] not in hpolicy:
            probs = []
            count = []
            for action in actions:
                try:
                    count.append(trajectory[(key[0], action)])
                except:
                    count.append(0)
            total_cnt = sum(count)
            # probs = [0.6**count[int(action.numpy())]*(1-0.6)**(total_cnt-count[int(action.numpy())]) for action in actions]
            probs = [0.6**count[action]*(1-0.6)**(total_cnt-count[action]) for action in actions]
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hpolicy[key[0]] = probs
    
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    for iteration in range(args.load_from+1, args.load_from + args.num_iterations + 1):
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
                # _, logprob_ref, _, _ = agent_ref.get_action_and_value(next_obs, action=action)
                values[step] = value.flatten()
            
            # print(non_score_reward)
            the_actions = action.cpu().numpy()
            # TRY NOT TO MODIFY: execute the game and log data.
            shaping_reward = []
            for i in range(args.num_envs):
                unwrapped_env = envs.envs[i].unwrapped
                envstate = envs.observations[i] # the unwrapped env might not have a flat observation space
                # envstate_Update = [2,3,7,7,4,4,3,3]                
                curr_agent_pos = envstate[:2]
                
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            key = tuple(next_obs.flatten())
            if key in hpolicy:
                hprobs = hpolicy[key]
                actions[step] = action
                logprobs[step] = logprob
                logprobs_ref[step] =  np.log(hprobs[action])
                # kl_penalty_factor = 2 # 2 for Milk and 0.25 for Drive based on Moral paper https://github.com/kristery/EthicsShaping/blob/master/Drive/hsarsa_n.py
                kl_penalty_factor = 2
                with torch.no_grad():
                    lp_finetune = nn.functional.log_softmax(logprobs[:step+1], dim=0)
                    lp_ref = nn.functional.log_softmax(logprobs_ref[:step+1], dim=0)
                    kl = kl_loss(lp_finetune,lp_ref).detach().numpy()
                    # kl_rohit = kl_div(lp_ref,lp_finetune)                
                    writer.add_scalar(f"charts/episodic_kl_divergence", kl, global_step)
                print('kl divergence: ',kl)
                non_score_reward = -(kl_penalty_factor * kl)
        
                reward = reward + non_score_reward
           
            # logText = f"{infos['metric1'][0]} {infos['metric1'][0]} {infos['metric5'][0]} Reward {reward} Kl: {kl}"
            # print(logText)
            
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
