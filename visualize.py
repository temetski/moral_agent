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
import csv

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
        if indices is None:
            indices = list(range(len(frames)))
        for i, frame in enumerate(frames):
            if i in indices:
                clear_output(wait=True)
                print(f"Timestep: {i + 1}")
                # print(f"State: {frame['state']}")
                # print(f"Action: {frame['action']}")
                # print(f"Reward: {frame['reward']}")
                print(f"Number of collisions: {frame['metric_1']}")
                print(f"Number of hit cats: {frame['metric_2']}")
                sleep(dt)     
    else:
        if indices is None:
            indices = list(range(len(frames)))
        for i, frame in enumerate(frames):
            if i in indices:
                clear_output(wait=True)
                print(frame['frame'])
                print(f"Timestep: {frame['timestep']}")
                print(f"State: {frame['state']}")
                print(f"Action: {frame['action']}")
                print(f"Reward: {frame['reward']}")
                print(f"{frame['metric_1_name']}: {frame['metric_1']}")
                print(f"{frame['metric_2_name']}: {frame['metric_2']}")
                sleep(dt)
            
def run(config):    
    run_name = f"{config.env_id.replace(':','.')}__{config.exp_name}__{config.seed}__moral"
    env = gym.make(config.env_id, render_mode='ansi', validate=True)
    env = gym.wrappers.FlattenObservation(env)
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

        metrics = env.log()
        # Put each rendered frame into dict for animation
        steps += 1

        frames.append({
            'timestep': steps,
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward,
            'metric_1_name' : metrics['metric1'][0],
            'metric_2_name' : metrics['metric2'][0],
            'metric_1' : metrics['metric1'][1],
            'metric_2' : metrics['metric2'][1]
            }
        )
        next_obs, next_done = torch.Tensor(state).to(device), torch.Tensor(done).to(device)
        
    env.close()
    return frames


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo", type=str)
    # parser.add_argument("--env_id", default="environments.drive:Driving", type=str)  #"environments.drive:Driving"
    parser.add_argument("--env_id", default="environments.milk:FindMilk-v4", type=str)  #"environments.drive:Driving"
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--model_path", default=f"runs/FindMilk-v4__ppo__1__moral/kl_div/ppo_1800.cleanrl_model", type=str) #Moral model with both ishuman_p and ishuman_n as False. Else case reward = -20 * car_hit + 0.5 * (action == 0)
    # parser.add_argument("--model_path", default=f"runs/FindMilk-v4__ppo__1__1724503897/ppo.cleanrl_model", type=str)
    parser.add_argument("--capture_video", default=False, type=bool)
    parser.add_argument("--debug_llm", action="store_true")
    return parser


if __name__ == '__main__':    
    parser = argparser()
    config = parser.parse_args()


    stats = []

    for i in range(50):
        frames = run(config)
        stats.append(frames[-1])
        config.seed += 1
    
    print_frames(config.env_id, frames, dt=0.01, indices=[0, len(frames)-1])

    timesteps = [i['timestep'] for i in stats]
    metric_1 = [i['metric_1'] for i in stats]
    metric_2 = [i['metric_2'] for i in stats]
    print(f'Timesteps: {np.mean(timesteps)} +- {np.std(timesteps)}')
    print(f'{stats[0]["metric_1_name"]}: {np.mean(metric_1)} +- {np.std(metric_1)}')
    print(f'{stats[0]["metric_2_name"]}: {np.mean(metric_2)} +- {np.std(metric_2)}')
    print(f'{stats[0]["metric_1_name"]}: {metric_1}')
    print(f'{stats[0]["metric_2_name"]}: {metric_2}')        

    longest_idx = np.argmax(timesteps)
    print(stats[longest_idx]['frame'])

