import gymnasium as gym


# env = gym.make('environments.milk:FindMilk', render_mode='ansi', max_episode_steps=1500)
env = gym.make('environments.drive:Driving', render_mode='ansi', max_episode_steps=1500)

state = env.reset()
done = env.done
steps = 0
frames = []
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    steps += 1
    
env.close()

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
        sleep(dt)
        
print_frames(frames, dt=0.01)