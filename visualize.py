import gymnasium as gym
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from algorithms.ppo import Args, Agent, make_env
args = tyro.cli(Args)
run_name = f"{args.env_id.replace(':','.')}__{args.exp_name}__{args.seed}__moral"
# env = gym.make('environments.milk:FindMilk', render_mode='ansi', max_episode_steps=1500)
# env = gym.make('environments.drive:Driving', render_mode='ansi', max_episode_steps=1500)
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(1)],
)
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
agent = Agent(envs).to(device)
LOADPATH = "runs/environments.milk.FindMilk__ppo__1__1723014023/ppo.cleanrl_model" #Remove hardcode folderpath
agent.load_state_dict(torch.load(LOADPATH))
# envs.envs[0].render_mode='ansi'
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
# done = False
done = torch.zeros(1).to(device)
steps = 0
frames = []
itr = 0
while not done:
    # action = env.action_space.sample()
    action, logprob, _, value = agent.get_action_and_value(next_obs)
    state, reward, terminated, truncated, info = envs.envs[0].step(action)
    done = np.logical_or(terminated, truncated)
    itr=itr+1
    neg_passed, pos_passed = envs.envs[0].log()
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': envs.envs[0].render(),
        'state': state,
        'action': action,
        'reward': reward,
        'passed non-crying babies' : neg_passed,
        'passed crying babies' : pos_passed
        }
    )
    next_obs, next_done = torch.Tensor(state).to(device), torch.Tensor(done).to(device)
    
    steps += 1
#     if steps%10==0:
#         print(env.get_scenario_prompt())
#         print(env.state_as_text())
envs.envs[0].close()

from IPython.display import clear_output
from time import sleep

def print_frames(frames, dt=0.1):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print(f"Passed non-crying babies: {frame['passed non-crying babies']}")
        print(f"Passed crying babies: {frame['passed crying babies']}")
        sleep(dt)
        
print_frames(frames, dt=0.01)