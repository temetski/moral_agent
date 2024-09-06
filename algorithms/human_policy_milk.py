import numpy as np
import pickle
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
from ppo import Args, Agent, make_env
import gymnasium as gym
import torch
import numpy as np
np.random.seed(42)


def manhattan_distance(point1, point2):   
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

env = gym.make("environments.milk:FindMilk-v4", render_mode='ansi', validate=True, seed=42)
env = gym.wrappers.FlattenObservation(env)
# Mimic SyncVectorEnv for cleanrl's PPO
env.single_action_space = env.action_space
env.single_observation_space = env.observation_space


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)

rewards = []
trajectory = {}

for cnt in range(1000):
    state, _ = env.reset()
    while True:
        babies_state = state[4:]
        agent_pos = state[:2]
        crying_baby_pos = babies_state[:2]
        sleeping_baby_pos = babies_state[2:]
        milk_pos = state[2:4]
        
        max_distance = 24
        probs = []
        ### human policy ###
        for a in env.actions:
            next_agent_pos = env.next_pos(agent_pos[0],agent_pos[1],a)
            dist_agent_cryingBabies = manhattan_distance(next_agent_pos,crying_baby_pos)
            dist_agent_milk = manhattan_distance(next_agent_pos,milk_pos)
            #Based on human intuition. Human will give high probability to next agent position if it brings closer to the crying babies and not step on sleeping babies. 
            if next_agent_pos in env.sleep_positions:
                probs.append(1)
            elif dist_agent_milk==0:
                probs.append(20) # if 
            else:
                if max_distance > dist_agent_cryingBabies:
                    prob = ((max_distance - dist_agent_cryingBabies)/max_distance)*20
                    probs.append(prob) # 20 if crying baby is in surroundings, else decays with distance
                else:
                    probs.append(5) # why 5 if crying baby beyond max distance?

        # normalization
        total = sum(probs)
        probs = [p/total for p in probs]
        
        #select action with max probabilities. In case more than one max value, choose by random
        # max_prob = np.max(probs)
        # max_indices = np.where(probs == max_prob)[0]
        # # action_1 = np.random.choice(max_indices) 
        action = np.random.choice(4, 1, p=probs)[0]
        
        state_tuple = tuple(state)

        try:
            trajectory[(state_tuple, action)] += 1
        except:
            trajectory[(state_tuple, action)] = 1

        # state, reward, done = fm.step(action)
        state, reward, terminations, truncations, infos = env.step(action)
        rewards.append(reward)
        done = np.logical_or(terminations, truncations)
        if done:
            break

    logdata = env.log()    
    crying_babies, sleeping_babies = logdata['metric1'][1], logdata['metric2'][1]
    if cnt % 10 == 0:
        print(f'episode: {cnt}, total reward: {reward}, crying_babies: {crying_babies}, sleeping_babies: {sleeping_babies}')

path = f"runs/human_policy/hpolicy_milk.pkl"
with open(path, 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)