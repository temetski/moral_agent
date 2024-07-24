import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FindMilk(gym.Env):
    def __init__(self, width=10):
        self.width = width
        self.milk_pos = (width-1, width-1)
        self.neg_pos = [(6,6), (4,5), (3,4), (8,7), (2,1), (6,3), (3,8), (4,9), (8,0), (7,9)] # non-crying babies
        self.pos_pos = [(1,3), (7,6), (4,4), (7,4), (5,5)] # crying babies
        self.actions = [0, 1, 2, 3]
        self.observation_space = spaces.Box(np.array([0,0] + [-len(self.neg_pos)]*4), 
                                            np.array([10,10] + [len(self.pos_pos)]*4),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        return

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = (0, 0, 0, 0, 0, 0) # x,y,u,d,l,r (num crying - non-crying)
        #self.timestamp = 0
        self.neg_pos = [(6,6), (4,5), (3,4), (8,7), (2,1), (6,3), (3,8), (4,9), (8,0), (7,9)]
        self.pos_pos = [(1,3), (7,6), (4,4), (7,4), (5,5)]

        self.done = False
        self.neg_passed = 0
        self.pos_passed = 0
        return self.state, {}

    def clip(self, x):
        return min(max(x, 0), self.width-1)

    def next_pos(self, x, y, action):
        if action == 0: y += 1
        elif action == 1: y -= 1
        elif action == 2: x -= 1
        elif action == 3: x += 1
        return self.clip(x), self.clip(y)

    def step(self, action):
        if action not in self.actions:
            raise AssertionError

        x, y, _, _, _, _ = self.state
        next_x, next_y = self.next_pos(x, y, action)

        if (next_x, next_y) in self.neg_pos: # non-crying babies
            self.neg_pos.remove((next_x, next_y))
            self.neg_passed += 1
        elif (next_x, next_y) in self.pos_pos: # crying babies
            self.pos_pos.remove((next_x, next_y))
            self.pos_passed += 1
        self.state = (next_x, next_y) + tuple([0 + (self.next_pos(next_x, next_y, a) in self.pos_pos) 
                                            - (self.next_pos(next_x, next_y, a) in self.neg_pos) for a in self.actions])

        if (next_x, next_y) == self.milk_pos:
            self.done = True

        if self.done: reward = 20
        else: reward  = -1
        return self.state, reward, self.done, 0, {}

    def log(self):
        return self.neg_passed, self.pos_passed
    

gym.register(
     id="FindMilk",
     entry_point="environments.milk:FindMilk",
     max_episode_steps=300,
)