from typing import Optional, Any
import numpy as np
import gymnasium as gym
import logging
from .GameEngine import GameEngine

OPTIONS_ENV_PIECE_COLOUR = 'OPTIONS_ENV_PIECE_COLOUR'


class ConnectFourEnv(gym.Env):
    GREEN_PIECE_STATE = np.array([1, 0])
    RED_PIECE_STATE = np.array([0, 1])
    BLANK_PIECE_STATE = np.array([0, 0])


    STONE_STATE_NUM = 2

    def __init__(self):
        # The size of the grid (6x7 by default). i.e. 6 rows, 7 columns
        self.game_engine = GameEngine()
        self.board_state = self.game_engine.create_board()
        self.board_obs_state = np.zeros((ConnectFourEnv.STONE_STATE_NUM,GameEngine.ROW_COUNT, GameEngine.COLUMN_COUNT))

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Box(0, 1, shape=(ConnectFourEnv.STONE_STATE_NUM, self.game_engine.ROW_COUNT,self.game_engine.COLUMN_COUNT), dtype=np.float32)   # [x, y] coordinates

        # Define what actions are available (number of columns)
        self.action_space = gym.spaces.Discrete(self.game_engine.COLUMN_COUNT)

        # default game state
        self.game_env_piece_colour = GameEngine.RED_PIECE
        self.game_non_env_piece_colour = GameEngine.GREEN_PIECE
        self.game_n_step = 0

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = { OPTIONS_ENV_PIECE_COLOUR : GameEngine.RED_PIECE }):
        self.board_state = self.game_engine.create_board()

        self.game_env_piece_colour = options.get(OPTIONS_ENV_PIECE_COLOUR, GameEngine.RED_PIECE)

        if self.game_env_piece_colour == GameEngine.RED_PIECE:
            self.game_non_env_piece_colour = GameEngine.GREEN_PIECE
        else:
            self.game_non_env_piece_colour = GameEngine.RED_PIECE

        self.game_n_step = 0

        info = { OPTIONS_ENV_PIECE_COLOUR: self.game_env_piece_colour}

        return self._get_obs(), info


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        # board_obs_state[0] == Player's piece board state,
        # board_obs_state[1] == Env's piece board state
        self.board_obs_state[0, :, :] = (self.board_state == self.game_non_env_piece_colour)
        self.board_obs_state[1, :, :] = (self.board_state == self.game_env_piece_colour)

        return self.board_obs_state

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.game_n_step = self.game_n_step + 1

        reward = -0.01
        terminated = False
        truncated = False
        info = { 'step' : self.game_n_step}
        observation = self.board_state

        if self.game_n_step > 10 :
            terminated = True
            reward = 1

        # logging.info(f'get step hahahah. obs : {type(observation)},  obs shape : {observation.shape}')

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        logging.info(f'obs : {self.board_state}')
