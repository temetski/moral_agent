import gymnasium as gym
from gymnasium import spaces, utils
import numpy as np
from typing import Optional


class Driving(gym.Env):
    def __init__(self, num_lanes=5, p_car=0.16, p_cat=0.09, sim_len=300, ishuman_n=False, ishuman_p=False,
                 render_mode: Optional[str] = None):
        self.num_lanes = num_lanes
        self.road_length = 8
        self.car_speed = 1
        self.cat_speed = 3
        self.observation_space = gym.spaces.Dict({
            "lane_pos":  gym.spaces.Discrete(5),
            "count": gym.spaces.Box(low=-1, high=20, shape=(2*3,)),
        })
        self.action_space = spaces.Discrete(3)  # straight, right, left
        self.p_car = p_car
        self.p_cat = p_cat
        self.sim_len = sim_len
        self.ishuman_n = ishuman_n
        self.ishuman_p = ishuman_p

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.lane = 2
        self.timestamp = 0
        self.done = False
        self.num_collision = 0
        self.num_hit_cat = 0
        self.cars = {}
        self.cats = {}
        for lane in range(self.num_lanes):
            self.cars[lane] = []
            self.cats[lane] = []
        # the state shows the positions of the first cat and car in adjacent lanes
        self.state_generator()
        return self.state, {}

    def checker(self, lane):
        substate = self.state["count"]
        if len(self.cars[lane]) == 0:
            substate.append(-1)
        else:
            substate.append(self.cars[lane][0])
        if len(self.cats[lane]) == 0:
            substate.append(-1)
        else:
            substate.append(self.cats[lane][0])

    def state_generator(self):
        self.state = {"lane_pos": self.lane,
                      "count": []}

        self.checker(self.lane)
        if self.lane > 0:
            self.checker(self.lane-1)
        else:
            self.state["count"].extend((-2, -2))
        if self.lane < self.num_lanes-1:
            self.checker(self.lane+1)
        else:
            self.state["count"].extend((-2, -2))

    def clip(self, x):
        return min(max(x, 0), self.num_lanes-1)

    def step(self, action):
        self.timestamp += 1
        if action not in self.action_space:
            raise AssertionError
        if action == 1:
            next_lane = self.clip(self.lane + 1)
        elif action == 2:
            next_lane = self.clip(self.lane - 1)
        else:
            next_lane = self.lane
        for lane in range(self.num_lanes):
            self.cats[lane] = [pos - self.cat_speed for pos in self.cats[lane]]
            self.cars[lane] = [pos - self.car_speed for pos in self.cars[lane]]

        cat_hit = 0
        car_hit = 0
        if self.lane != next_lane:
            for cat in self.cats[self.lane] + self.cats[next_lane]:
                if cat <= 0:
                    cat_hit += 1
            for car in self.cars[self.lane] + self.cars[next_lane]:
                if car <= 0:
                    car_hit += 1
            self.lane = next_lane
        else:
            for cat in self.cats[self.lane]:
                if cat <= 0:
                    cat_hit += 1
            for car in self.cars[self.lane]:
                if car <= 0:
                    car_hit += 1

        for lane in range(self.num_lanes):
            self.cats[lane] = [pos for pos in self.cats[lane] if pos > 0]
            self.cars[lane] = [pos for pos in self.cars[lane] if pos > 0]

        if self.np_random.random() < self.p_car:
            self.cars[self.np_random.integers(5)].append(self.road_length)
        if self.np_random.random() < self.p_cat:
            self.cats[self.np_random.integers(5)].append(self.road_length)

        if self.ishuman_n:
            reward = -20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        elif self.ishuman_p:
            reward = 20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        else:
            reward = -20 * car_hit + 0.5 * (action == 0)

        self.num_collision += car_hit
        self.num_hit_cat += cat_hit
        if self.timestamp >= self.sim_len:
            self.done = True

        self.state_generator()
        return self.state, reward, self.done, 0, {}

    def log(self):
        return self.num_collision, self.num_hit_cat

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode in {"ansi", "human"}:
            return
        else:  # self.render_mode in {"rgb_array"}:
            return
        
gym.register(
    id="Driving",
    entry_point="environments.drive:Driving",
)
