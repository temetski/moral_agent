from environments import milk

env = milk.FindMilk()

state = env.reset()
done = env.done
steps = 0
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # if reward == -10:
    #     penalties += 1
    
    # Put each rendered frame into dict for animation
    # frames.append({
    #     # 'frame': env.render(),
    #     'state': state,
    #     'action': action,
    #     'reward': reward
    #     }
    # )

    steps += 1
    
    
print("Timesteps taken: {}".format(steps))
# print("Penalties incurred: {}".format(penalties))