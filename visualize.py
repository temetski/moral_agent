import gymnasium as gym
import torch

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from algorithms.ppo import Agent, make_env
from IPython.display import clear_output
from time import sleep
import argparse


def print_frames(env_id,frames, dt=0.1):
    if env_id=="environments.drive:Driving":
        #visualize Driving
        print("No visualization yet")
    else:
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
            
def run(config):    
    # args = tyro.cli(Args)
    run_name = f"{config.env_id.replace(':','.')}__{config.exp_name}__{config.seed}__moral"
    # env = gym.make('environments.milk:FindMilk', render_mode='ansi', max_episode_steps=1500)
    # env = gym.make('environments.drive:Driving', render_mode='ansi', max_episode_steps=1500)
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, i, config.capture_video, run_name) for i in range(config.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    agent = Agent(envs).to(device)

    agent.load_state_dict(torch.load(config.model_path))
    # envs.envs[0].render_mode='ansi'
    next_obs, _ = envs.reset(seed=config.seed)
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
    envs.envs[0].close()
    print_frames(config.env_id,frames, dt=0.01)
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo", type=str)
    parser.add_argument("--env_id", default="environments.drive:Driving", type=str)  #"environments.drive:Driving"
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model_path", default="runs/PreTrainedDriveModel/ppo.cleanrl_model", type=str) #PreTrainedDriveModel
    parser.add_argument("--capture_video", default=False, type=bool)

    config = parser.parse_args()
    run(config)