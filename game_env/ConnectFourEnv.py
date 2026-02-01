from typing import Optional, Any
import numpy as np
import gymnasium as gym
import logging
import random
from .GameEngine import GameEngine
from. import Renderer

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
        self.action_space = gym.spaces.Discrete(self.game_engine.COLUMN_COUNT)        # default game state
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

        is_env_game_win = False
        is_env_valid_move = False
        is_non_env_game_win = False
        is_non_env_valid_move = False
        is_draw = False

        logging.info(f'---- new step -----')
        logging.info(f'non env action : {action}')

        ## non env move
        valid_locations = GameEngine.get_valid_locations(self.board_state)
        if len(valid_locations) == 0 :  ### no possible move
            is_draw = True
            reward = 0.5
            terminated = True
        else:
            is_non_env_valid_move, is_non_env_game_win = GameEngine.try_drop_piece(self.board_state, action, self.game_non_env_piece_colour)
            if not is_non_env_valid_move:
                terminated = True
                reward = -1.0
            elif is_non_env_game_win: ## non env win the game
                    terminated = True
                    reward = 1.0
            else:
                ## non env completed the move, now it is the env's turn to response
                # env_action = column = random.choice(valid_locations)
                valid_locations = GameEngine.get_valid_locations(self.board_state)
                if len(valid_locations) == 0:  ### no possible move
                    is_draw = True
                    reward = 0.5
                    terminated = True
                else:
                    env_action_col = random.choice(valid_locations)
                    # logging.info(f'env action : {env_action_col}')
                    is_env_valid_move, is_env_game_win = GameEngine.try_drop_piece(self.board_state, env_action_col, self.game_env_piece_colour)

                    if not is_env_valid_move:
                        terminated = True
                        reward = 0.5

                    elif is_env_game_win: ## env win the game
                        terminated = True
                        reward = -1.0

                    else:
                        if self.game_n_step > 20 :
                            terminated = True
                            reward = 1
                        else:
                            reward = -0.01

        observation = self._get_obs()
        info = { 'step' : self.game_n_step, 'env_win': is_env_game_win, 'non_env_win' : is_non_env_game_win,
                 'env_valid_move' : is_env_valid_move, 'non_env_valid_move' : is_non_env_valid_move , 'is_draw' : is_draw }

        # logging.info(f'reward : {reward}, info : {info}')
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        Renderer.render(self.board_state, mode)
