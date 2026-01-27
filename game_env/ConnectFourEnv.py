import numpy as np
import gymnasium as gym


class ConnectFourEnv(gym.Env):

    def __init__(self, size: tuple= (6,7)):
        # The size of the grid (6x7 by default). i.e. 6 rows, 7 columns
        self.board_size = size

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Box(0, 1, shape=(self.board_size[0],self.board_size[1]), dtype=np.float32)   # [x, y] coordinates

        # Define what actions are available (number of columns)
        self.action_space = gym.spaces.Discrete(self.board_size[1])