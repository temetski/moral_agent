import numpy as np
import pickle
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import torch
import numpy as np
np.random.seed(42)

env = gym.make("environments.drive:Driving", seed=42, render_mode='ansi', validate=True, ishuman_p=True)
env = gym.wrappers.FlattenObservation(env)
env.single_action_space = env.action_space
env.single_observation_space = env.observation_space


# Function to safely convert arrays and lists to hashable tuples
def to_hashable(item):
    """Converts numpy arrays or lists to tuples, leaves other types unchanged."""
    if isinstance(item, np.ndarray):
        return tuple(item)
    elif isinstance(item, (list, tuple)):
        return tuple(item)
    else:
        return item  # Return as is if it's already an immutable type (e.g., int)
    
trajectory = {}
episode_rewards = []
collisions = []
cat_hits = []
actions = range(3)
for cnt in range(10000):
    state, _ = env.reset()
    rewards = 0.
    prev_pair = None
    prev_reward = None
    frame = 0
    Q = {}
    while True:
        frame += 1
        probs = []
        for action in actions:
            try: 
                probs.append(np.e**(Q[(tuple(state), action)]/0.7)) #the temperature parameter for Q learning policy (default: 0.7)'
            except:
                Q[(tuple(state), action)] = np.random.randn()
                probs.append(np.e**(Q[(tuple(state), action)]/0.7)) #the temperature parameter for Q learning policy (default: 0.7)'

        total = sum(probs)
        probs = [p / total for p in probs]
        
        action = np.random.choice(actions, p=probs)
        # print(probs, state, action)

        # ethical_state = (state[1], state[3], state[5]) #state[2] is cat for center lane, state[4] cat for left lane, state[6] cat for right lane. This for old code. We revereted to match the one hot encoding
        ethical_state = tuple(state) 
        if cnt > 600: #after this number it begin to record trajectories
            try:
                trajectory[(ethical_state, action)] += 1
            except:
                trajectory[(ethical_state, action)] = 1


        if prev_pair is not None:
            # current_key = (tuple(state), action)
            Q[prev_pair] = Q[prev_pair] + 0.1 * (prev_reward + 0.99 * Q[(tuple(state), action)] - Q[prev_pair])
        next_state, reward, done, truncations, infos = env.step(action)

        prev_pair = (tuple(state), action)
        prev_reward = reward
        rewards += reward
        if done:
            Q[prev_pair] = Q[prev_pair] + 0.1 * (prev_reward - Q[prev_pair])
            break
        state = next_state
    logdata = env.log()    
    collision, cat_hit = logdata['metric1'][1], logdata['metric2'][1]
    collisions.append(collision)
    cat_hits.append(cat_hit)
    episode_rewards.append(rewards)
    
    if cnt % 10 == 0:
        print(f'episode: {cnt}, frame: {frame}, total reward: {rewards}, collision: {collision}, grandmas: {cat_hit}')

print(np.mean(collisions[-300:]), np.mean(cat_hits[-300:]))
with open('runs/human_policy/hpolicy_drive.pkl', 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)