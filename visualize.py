import gymnasium as gym
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from algorithms.ppo import Agent, make_env
from IPython.display import clear_output
from time import sleep
import argparse
from algorithms.llm_moral import call_llm_with_state_action,create_llm_env,few_shot_prompt_training

NUM_MORAL = 5

credences = np.zeros((5, NUM_MORAL))
# Set the diagonal elements
for i in range(NUM_MORAL):
    credences[i, i] = 1
    
api_key = os.environ.get("OPENAI_API_KEY", "none")
model = create_llm_env(api_key)
final_prompt = few_shot_prompt_training()

def print_frames(env_id, frames, dt=0.1, indices=None):
    if "Driving" in env_id:
        #visualize Driving
        print("No visualization yet")
    else:
        if indices is None:
            indices = list(range(len(frames)))
        for i, frame in enumerate(frames):
            if i in indices:
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
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(config.env_id, i, config.capture_video, run_name) for i in range(config.num_envs)],
    # )
    env = gym.make(config.env_id, render_mode='ansi')
    # Mimic SyncVectorEnv for cleanrl's PPO
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    agent = Agent(env).to(device)

    agent.load_state_dict(torch.load(config.model_path))

    next_obs, _ = env.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    done = False
    steps = 0
    frames = []
    itr = 0

    while not done:
        # action = env.action_space.sample()
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        if config.debug_llm:
            print(env.render())
            state_text, action_text = env.state_as_text()
            actionsets = [frozenset([str(k)]) for k in env.action_mapper.keys()] #TODO: review str casting 
            scenario_prompt = env.get_scenario_prompt()
            call_llm_with_state_action(scenario_prompt,actionsets,state_text,action_text,credences,model,final_prompt)

        state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        itr=itr+1
        neg_passed, pos_passed = env.log()
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward,
            'passed non-crying babies' : neg_passed,
            'passed crying babies' : pos_passed
            }
        )
        next_obs, next_done = torch.Tensor(state).to(device), torch.Tensor(done).to(device)
        
        steps += 1
    env.close()
    return frames

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo", type=str)
    parser.add_argument("--env_id", default="environments.drive:Driving", type=str)  #"environments.drive:Driving"
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model_path", default="runs/PreTrainedDriveModel/ppo.cleanrl_model", type=str) #PreTrainedDriveModel
    parser.add_argument("--capture_video", default=False, type=bool)
    parser.add_argument("--debug_llm", action="store_true")

    config = parser.parse_args()
    frames = run(config)
    print_frames(config.env_id, frames, dt=0.01, indices=[0, len(frames)-1])
