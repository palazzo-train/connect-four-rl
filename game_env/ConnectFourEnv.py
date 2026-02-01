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


    def _check_must_win_location(self, action, piece, valid_locations):
        taken_right_move = False
        exist_must_win = False

        # logging.info(f' check must win, valid location : {valid_locations}')
        ###
        ### check must win col location
        must_win_locations = []
        for trial_location in valid_locations:
            is_this_move_win = GameEngine.trial_drop_piece_from_top(self.board_state, trial_location, piece)
            # logging.info(f'  is this move win, : {is_this_move_win}, trial_location : {trial_location}')
            if is_this_move_win:
                if action == trial_location:
                    ### non env makes the right move, can return
                    is_non_env_game_win = GameEngine.trial_drop_piece_from_top(self.board_state, action, piece)
                    taken_right_move = True
                    exist_must_win = True
                    # logging.info(f'  taken must win, valid location : {action}')
                    return exist_must_win, taken_right_move

                must_win_locations.append(trial_location)

        # logging.info(f' check must win, must win location: {must_win_locations}, len : {len(must_win_locations)}')
        # logging.info(f' sddd action in must_win: {not (action in must_win_locations)}')
        ### if there exists must-win location but non env player not take that move
        if (len(must_win_locations) > 0) and (not (action in must_win_locations)):
            exist_must_win = True
            taken_right_move = False
            return exist_must_win, taken_right_move

        return exist_must_win, taken_right_move

    def _non_env_move(self,action):
        valid_locations = GameEngine.get_valid_locations(self.board_state)
        is_non_env_game_win = False
        terminated = False
        is_draw = False
        reward = 0.0

        if len(valid_locations) == 0 :  ### no possible move
            is_draw, reward, terminated = True, 0.7, True  ## draw
        elif not GameEngine.is_valid_location(self.board_state, action):
            # logging.info(f'invalid move by non env , action: {action}')
            terminated , reward = True, -1.0  ## invalid move
        else:
            ### check must-win col location
            exist_must_win, taken_right_move = self._check_must_win_location(action, self.game_non_env_piece_colour, valid_locations)
            # logging.info(f'exist must win : {exist_must_win},  taken_right_move : {taken_right_move}')
            if exist_must_win:
                terminated = True
                reward = 1.0 if taken_right_move else -1.0

                if taken_right_move:
                    is_non_env_game_win = GameEngine.drop_piece_from_top(self.board_state, action, self.game_non_env_piece_colour)

                return terminated, reward, is_draw, is_non_env_game_win

            ###
            ### check must-defense col location, i.e. must-win of env_piece_colour
            exist_must_win, taken_right_move= self._check_must_win_location(action, self.game_env_piece_colour, valid_locations)
            if exist_must_win and not taken_right_move:
                terminated, reward = True, -1.0  ## does not take must-defense move
                return terminated, reward, is_draw, is_non_env_game_win

            ### taken defensive move, continue
            GameEngine.drop_piece_from_top(self.board_state, action, self.game_non_env_piece_colour)

        return terminated, reward, is_draw, is_non_env_game_win


    def _env_move_decision(self, valid_locations):
        env_action_col = random.choice(valid_locations)
        return env_action_col

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

        logging.info(f'---- new step ----- , step number : {self.game_n_step}')
        logging.info(f'non env action : {action}')


        ## non env move
        terminated, reward, is_draw , is_non_env_game_win = self._non_env_move(action)
        if not terminated:
            ### response by env. env's turn to move
            valid_locations = GameEngine.get_valid_locations(self.board_state)
            if len(valid_locations) == 0:  ### no possible move, draw
                is_draw , reward , terminated = True, 0.7, True
            else:
                ### env's to pick a move
                env_action_col = self._env_move_decision(valid_locations)
                is_env_game_win = GameEngine.drop_piece_from_top(self.board_state, env_action_col, self.game_env_piece_colour)

                if is_non_env_game_win:
                    reward ,terminated = -1.0, True

                ## new valid location after env response
                valid_locations = GameEngine.get_valid_locations(self.board_state)
                if len(valid_locations) == 0:  ### no possible move, draw
                    is_draw, reward, terminated = True, 0.7, True

        observation = self._get_obs()
        info = { 'step' : self.game_n_step, 'env_win': is_env_game_win, 'non_env_win' : is_non_env_game_win,
                 'env_valid_move' : is_env_valid_move, 'non_env_valid_move' : is_non_env_valid_move , 'is_draw' : is_draw }

        # logging.info(f' observation : \n{observation}')
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        Renderer.render(self.board_state, mode)
