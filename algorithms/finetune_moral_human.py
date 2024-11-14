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


kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)
    
## OVERRIDES
@dataclass
class FineTuneArgs(Args):
    num_steps: int = 128 # note it is 64 for Milk
    total_timesteps: int = 1000*num_steps
    num_envs: int = 1
    update_epochs: int = 8
    anneal_lr: bool = False
    # load_model: str = "models/Driving_42/base.cleanrl_model"
    load_model: str = "models/FindMilk-v4_42/base.cleanrl_model"
    # load_model: str = "runs/FindMilk-v4__ppo__1__1724503897/ppo.cleanrl_model" #The Milk base model that gave us good result. KL factor of 2
    load_model_ref: str = None#"runs/Driving__ppo__1__1724832763/ppo.cleanrl_model"
    load_from: int = 0
    write_to_csv: bool = True

kwargs = {'validate': True,
          'ishuman_p': False,
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
    run_name = f"{env_id}_{args.seed}"
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
    writer = SummaryWriter(f"models/{run_name}/RLHF/", filename_suffix="drive")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    agent.critic = agent.reset_critic(envs).to(device) # why? 
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
   
            
    #Load Human Policy 
    hpolicy = {}
    actions = range(envs.single_action_space.n)
    env_tag = 'milk' if "FindMilk" in env_id else 'drive'
    with open(f'runs/human_policy/hpolicy_{env_tag}.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    for (ethical_state, a) in trajectory.keys():
        if ethical_state not in hpolicy:
            probs = []
            count = []
            for action in actions:
                try:
                    count.append(trajectory[(ethical_state, action)])
                except:
                    count.append(0)
            total_cnt = sum(count)
            
            if total_cnt > 20:
                count = [p * 20 / total_cnt for p in count]
            
            total_cnt = sum(count)
            probs = [0.6**count[action]*(1-0.6)**(total_cnt-count[action]) for action in actions]
            # print(probs)
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hpolicy[ethical_state] = probs            
    
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    for iteration in range(args.load_from+1, args.load_from + args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        running_reward = []
        running_logprobs = []
        running_logprobs_ref = []
        frac_hpolicy = 0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                _, logprob_ref, _, _ = agent_ref.get_action_and_value(next_obs, action=action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            running_logprobs.append(logprob)
            running_logprobs_ref.append(logprob_ref)
            kl_penalty_factor = 2.5 # 2 for Milk and 0.25 for Drive based on Moral paper https://github.com/kristery/EthicsShaping/blob/master/Drive/hsarsa_n.py
            with torch.no_grad(): # tensors all on CPU
                lp_finetune = nn.functional.log_softmax(torch.Tensor(running_logprobs), dim=0)
                lp_ref = nn.functional.log_softmax(torch.Tensor(running_logprobs_ref), dim=0)
                kl = kl_loss(lp_finetune, lp_ref).detach().numpy()
            writer.add_scalar(f"charts/kl_div", kl, global_step)
            non_score_reward = -(kl_penalty_factor * kl)
            the_actions = action.cpu().numpy()
            # TRY NOT TO MODIFY: execute the game and log data.
            shaping_reward = []
            # if env_tag=='drive':
            #     ethical_state = tuple(next_obs.flatten()[1:6:2].tolist())
            # else:
            ethical_state = tuple(next_obs.flatten().tolist())
            for i in range(args.num_envs):
                unwrapped_env = envs.envs[i].unwrapped
                envstate = envs.observations[i] # the unwrapped env might not have a flat observation space
                if ethical_state in hpolicy:
                    hprobs = hpolicy[ethical_state]
                    RLHF_reward = hprobs[action]
                    frac_hpolicy += 1
                else: RLHF_reward = 0
            next_obs, env_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            # reward += non_score_reward
            # reward = 0
            # if ethical_state in hpolicy:
            reward = non_score_reward + RLHF_reward
            if any(next_done): # only works for num_envs==1
                # reward = non_score_reward + np.mean(running_reward)*any(terminations)
                running_reward = []
                running_logprobs = []
                running_logprobs_ref = []
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/finetune_return", reward, global_step)
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
        print("frac hpolicy:", frac_hpolicy/args.num_steps)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and (iteration%5==0 or iteration==args.num_iterations):
            model_path = f"models/{run_name}/RLHF/{args.exp_name}_{iteration}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    envs.close()
    writer.close()
