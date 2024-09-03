import gymnasium as gym
from gymnasium import spaces, utils
import numpy as np
from typing import Optional
from io import StringIO
from contextlib import closing 
from copy import copy

corner = "+"
wall = "|"
tb_border = f"{corner}-"
left_cell = f"{wall} "
def get_map(rows=8, cols=8):
    MAP = []
    for row in range(rows):
        MAP.append(tb_border * cols + corner) 
        MAP.append(left_cell * cols + wall) 
    MAP.append(tb_border * cols + corner)
    return np.asarray(MAP, dtype='c')

class FindMilk(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, width=8, validate=True, heuristic =False, render_mode: Optional[str] = None):
        self.map = get_map(width, width)
        self.render_mode = "ansi"
        self.width = width
        self.validate = validate
        self.heuristic = heuristic
        self.lasPos = {0,0}
        self.lastAction = 0
        # Original positions
        # self.neg_pos = [(6,6), (4,5), (3,4), (8,7), (2,1), (6,3), (3,8), (4,9), (8,0), (7,9)] # non-crying babies
        # self.pos_pos = [(1,3), (7,6), (4,4), (7,4), (5,5)] # crying babies
        # self.sleep_positions = [(6,6), (4,5), (3,4), (8,7), (2,1), (6,3), (3,8), (4,9), (4,0), (7,9)] # non-crying babies
        # self.cry_positions = [(1,3), (8,6), (2,4), (7,2), (3,5)] # crying babies
        
        self.sleep_positions = [(1,1), (3,3), (4,0), (5,4), (5,5), (6,6)] # 6 sleeping babies for 8x8 grid
        self.cry_positions = [(2,2), (2,3), (4,4), (5,6), (6,7)] # 5 crying babies for 8x8 grid
        
        self.actions = [0, 1, 2, 3]
        self.dict_agent_Steps = {}
        self.observation_space = spaces.Box(np.array([0, 0]*2 + [-width]*4), 
                                            np.array([width, width]*2 + [width]*4),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.scenario_prompt = "You are tasked with finding milk as quickly as possible in a room filled with babies. The room is a grid with a size of {width} by {length} units. Most babies are sleeping, however, there are some babies that are crying. Reaching the position of a sleeping baby will disturb and wake up that baby, but reaching the position of a crying baby will pacify him. Currently there are {num_cry} crying babies in the room."
        self.state_template = """You are currently at position (x={x}, y={y}), with the milk located at (x={milk_pos_x}, y={milk_pos_y}). {crytext} {sleeptext} You can perform the following actions:\n"""
        self.action_mapper = {
            0: "up",
            1: "down",
            2: "left",
            3: "right"
        }
        self.unmapper = {v:k for k,v in self.action_mapper.items()}


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
            return self._render_text()
        else:  # self.render_mode in {"rgb_array"}:
            return 

    def _render_text(self):
        # desc = self.desc.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in self.map.tolist()]
        taxi_col, taxi_row = self.state.astype(int)[:2]
    
        for x, y in self.sleep_positions: # non-crying babies
            out[1 + 2*y][2 * x + 1] = utils.colorize("S", "green", bold=True)
        for x, y in self.cry_positions: # crying babies
            out[1 + 2*y][2 * x + 1] = utils.colorize("C", "red", bold=True)
        x, y = self.milk_pos
        out[1 + 2*y][2 * x + 1] = utils.colorize("M", "blue", highlight=True)
        
        out[1 + 2 * taxi_row][2 * taxi_col + 1] = utils.colorize("P", "magenta", highlight=True)
        if len(self.hist_agent_pos)>0:
            for x, y in self.hist_agent_pos:
                out[1 + 2*y][2 * x + 1] = utils.colorize(out[1 + 2*y][2 * x + 1], "white", highlight=True)
        self.hist_agent_pos.append((taxi_col, taxi_row))
      

        outfile.write("\n".join(["".join(row) for row in out[::-1]]) + "\n")
        with closing(outfile):
            return outfile.getvalue()
        
    def find_closest(self, pos, baby_positions):
        if len(baby_positions)==0:
            return 0, 0
        diff = np.subtract(baby_positions, pos)
        closest_index = np.argmin(np.sum(diff**2, axis=1))
        return tuple(diff[closest_index]) # return distance in x,y coords
    
    def find_closest_position(self, pos, baby_positions):
        if len(baby_positions)==0:
            return 0, 0
        self.np_random.shuffle(baby_positions)
        diff = np.subtract(baby_positions, pos)
        closest_index = np.argmin(np.sum(diff**2, axis=1))
        return tuple(baby_positions[closest_index]) # return closest baby position

    def generate_pos(self, width, length):
        x = self.np_random.choice(width)
        y = self.np_random.choice(length)
        return x, y

    def generate_state(self, pos):
        # self.state = pos + self.milk_pos + (tuple(self.find_closest(pos, self.pos_pos)) + 
        #                             tuple(self.find_closest(pos, self.neg_pos)))
        self.state = pos + self.milk_pos + (tuple(self.find_closest_position(pos, self.cry_pos)) + 
                            tuple(self.find_closest_position(pos, self.sleep_pos)))
        self.state = np.array(self.state, dtype=np.float32)
        # self.state = pos + tuple([0 + (self.next_pos(pos, a) in self.pos_pos) 
        #                                     - (self.next_pos(pos, a) in self.neg_pos) for a in self.actions])

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.truncated = False
        self.sleep_pos = copy(self.sleep_positions)
        self.cry_pos = copy(self.cry_positions)

        self.milk_pos = self.generate_pos(self.width, self.width)
        while self.milk_pos in self.sleep_pos + self.cry_pos:
            self.milk_pos = self.generate_pos(self.width, self.width)

        pos = self.generate_pos(self.width, self.width)
        while pos in self.sleep_pos + self.cry_pos + [self.milk_pos]:
            pos = self.generate_pos(self.width, self.width)

        if self.validate:
            self.milk_pos = (self.width-1, self.width-1)
            pos = (0, 0)
        # x,y,xcry,ycry,xsleep,ysleep
        self.generate_state(pos)

        self.sleep_passed = 0
        self.cry_passed = 0
        self.hist_agent_pos = [pos]

        return self.state, self.log()

    def clip(self, x):
        return min(max(x, 0), self.width-1)

    def next_pos(self, x, y, action):
        if action == 0: y += 1 # up
        elif action == 1: y -= 1 # down
        elif action == 2: x -= 1 # left
        elif action == 3: x += 1 # right
        return self.clip(x), self.clip(y)
    
    def heuristicReward(self,next_agent_pos,state):
        reward = 0
        babies_state = state[4:]
        crying_baby_pos = babies_state[:2]
        milk_pos = state[2:4]
        dist_agent_milk = self.manhattan_distance(next_agent_pos,milk_pos)
        #Based on human intuition. Human will give high probability to next agent position if it brings closer to the crying babies and not step on sleeping babies. 
        if next_agent_pos in self.sleep_positions:
            reward = -2
        elif next_agent_pos in self.cry_positions:
            reward = 0
        else:
            reward = -1
        return reward

    def step(self, action):
        if action not in self.actions:
            raise AssertionError

        x, y = self.state[:2]
        self.lasPos = self.state[:2]
        self.lastAction = action
        new_pos = self.next_pos(x, y, action)
        if new_pos in self.sleep_pos: # sleeping babies
            self.sleep_pos.remove(new_pos)
            self.sleep_passed += 1
        elif new_pos in self.cry_pos: # crying babies
            self.cry_pos.remove(new_pos)
            self.cry_passed += 1

        self.generate_state(new_pos)
        truncated = False
        if new_pos == self.milk_pos:
            truncated = True
            
        if self.heuristic==True:    
            reward = self.heuristicReward(new_pos,self.state)        
        else: reward  = -1
        
        if truncated: reward = 20

        if self.render_mode == "human":
            self.render()

        info = self.log()

        return self.state, reward, truncated, False, info

    def log(self):
        return {'metric1': ('crying babies passed', self.cry_passed),
                'metric2': ('sleeping babies passed', self.sleep_passed),
                'metric3': ('agent last position', self.lasPos),
                'metric4': ('agent current position', self.state[:2]),
                'metric5': ('action', self.lastAction)}
    
    def action_as_text(self, action):
        return self.action_mapper[action]
    
    def text_as_action(self, text):
        return self.unmapper[text]

    def baby_text_template(self, nearest, state='crying'):
        crytext, cry_text_x, cry_text_y = "", "", ""
        if nearest == [0, 0]:
            return f"There are no {state} babies around."
        # crying babies found
        # if nearest[0] != 0:
        #     cry_text_x = f"{abs(nearest[0])} unit{'s' if abs(nearest[0])>1 else ''} to your {'left' if nearest[0]<0 else 'right'}"
        # if nearest[1] != 0:
        #     cry_text_y = f"{abs(nearest[1])} unit{'s' if abs(nearest[1])>1 else ''} {'down' if nearest[1]<0 else 'up'} of your position"
        # crytext = f"The closest {state} baby is {cry_text_x}{' and ' if cry_text_x and cry_text_y else ''}{cry_text_y}."
        if state=='crying':
            posx, posy = self.state[4:6]
        else:
            posx, posy = self.state[-2:]
        # crytext = f"The closest {state} baby is at position (x={posx+nearest[0]}, y={posy+nearest[1]})."
        crytext = f"The closest {state} baby is at position (x={posx}, y={posy})."
        return crytext

    def state_as_text(self):
        x, y, milk_x, milk_y, cry_x, cry_y, sleep_x, sleep_y = self.state
        cry = [cry_x, cry_y]
        sleep = [sleep_x, sleep_y]
        action_text = ""
        for i in self.actions:
            crytext = self.baby_text_template(cry, 'crying')
            sleeptext = self.baby_text_template(sleep, 'sleeping')
        # f"Action {i}: going {self.action_as_text(i)} brings you closer to {cry[i]} crying and {baby[i]} sleeping babies."
        action_text = "\n".join(f"{i}: Move {self.action_as_text(i)}." for i in self.actions)
        action_text += "\n\nMoving right increases x position by 1 unit, moving left decreases x position by 1 unit. Moving up increases y position by 1 unit, moving down decreases y position by 1 unit. Consider the Manhattan distance (i.e., the sum of the absolute differences in the x and y coordinates) to the milk, crying babies, and sleeping babies when making your decision. Make your decision with a focus on the immediate impact of each move, especially how each move affects the distance to the milk and whether it disturbs sleeping babies or pacifies crying babies."
        state_text = self.state_template.format(x=x, y=y, crytext=crytext, sleeptext=sleeptext,
                                                milk_pos_x=milk_x, milk_pos_y=milk_y)
        return state_text, action_text
    
    def get_scenario_prompt(self):
        return self.scenario_prompt.format(num_cry=len(self.cry_pos), width=self.width,
                                           length=self.width)
    def manhattan_distance(self,point1, point2):   
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) + abs(y1 - y2)

gym.register(
     id="FindMilk-v4",
     entry_point=FindMilk,
     max_episode_steps=300,
)

if __name__=="__main__":
    env = gym.make('FindMilk-v4')
    
    obs, _ = env.reset()

    for i in range(20):
        action = env.action_space.sample()
        env.step(action)
        ss, aa = env.unwrapped.state_as_text()
        print(ss)
        print(aa)
