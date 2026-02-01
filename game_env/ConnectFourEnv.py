from typing import Optional, Any
import numpy as np
import gymnasium as gym
import logging
import random
from .GameEngine import GameEngine
from. import Renderer

OPTIONS_ENV_PIECE_COLOUR = 'OPTIONS_ENV_PIECE_COLOUR'
OPTIONS_ENV_TRAINER_MODEL = 'OPTIONS_ENV_TRAINER_MODEL'


EXPLORATION_RATE = 0.5
ENV_FIRST_MOVE_RATE = 0.5

REWARD_NON_ENV_WIN = 0.6
REWARD_NON_ENV_INVALID_MOVE = -1.0
REWARD_DRAW = 0.5
REWARD_NON_ENV_LOSS = -0.6
REWARD_NON_ENV_MISSED_MUST_WIN = -0.6
REWARD_NON_ENV_MISSED_MUST_DEFENSE = -0.6
REWARD_WINNING_EARLY_EXTRA = 0.35
REWARD_LOSSING_EARLY_EXTRA = 0.35
REWARD_MULTI_OPPORTUNITY_EXTRA = 0.15
MAX_GAME_STEP = 21

class ConnectFourEnv(gym.Env):
    GREEN_PIECE_STATE = np.array([1, 0])
    RED_PIECE_STATE = np.array([0, 1])
    BLANK_PIECE_STATE = np.array([0, 0])


    STONE_STATE_NUM = 2

    def __init__(self, options: Optional[dict] = { OPTIONS_ENV_TRAINER_MODEL: None}):
        # The size of the grid (6x7 by default). i.e. 6 rows, 7 columns
        self.game_engine = GameEngine()
        self.board_state = self.game_engine.create_board()
        self.board_obs_state = np.zeros((ConnectFourEnv.STONE_STATE_NUM,GameEngine.ROW_COUNT, GameEngine.COLUMN_COUNT))
        self.board_obs_state_for_env_trainer_model = np.zeros((ConnectFourEnv.STONE_STATE_NUM,GameEngine.ROW_COUNT, GameEngine.COLUMN_COUNT))

        self.env_trainer_model = options.get(OPTIONS_ENV_TRAINER_MODEL, None)

        self.env_trainer_model = None


        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Box(0, 1, shape=(ConnectFourEnv.STONE_STATE_NUM, self.game_engine.ROW_COUNT,self.game_engine.COLUMN_COUNT), dtype=np.float32)   # [x, y] coordinates

        # Define what actions are available (number of columns)
        self.action_space = gym.spaces.Discrete(self.game_engine.COLUMN_COUNT)        # default game state
        self.game_env_piece_colour = GameEngine.RED_PIECE
        self.game_non_env_piece_colour = GameEngine.GREEN_PIECE
        self.game_n_step = 0

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = { OPTIONS_ENV_PIECE_COLOUR : GameEngine.RED_PIECE}):

        self.board_state = self.game_engine.create_board()
        self.game_env_piece_colour = options.get(OPTIONS_ENV_PIECE_COLOUR, GameEngine.RED_PIECE)

        if self.game_env_piece_colour == GameEngine.RED_PIECE:
            self.game_non_env_piece_colour = GameEngine.GREEN_PIECE
        else:
            self.game_non_env_piece_colour = GameEngine.RED_PIECE

        self.game_n_step = 0


        self.first_move = 'non env'
        ### does Env place the first move?
        random_number = random.random()
        if random_number < ENV_FIRST_MOVE_RATE:
            ### env first
            valid_locations = GameEngine.get_valid_locations(self.board_state)
            env_first_action_col = random.choice(valid_locations)
            GameEngine.drop_piece_from_top(self.board_state, env_first_action_col, self.game_env_piece_colour)
            self.first_move = 'env'

        info = { OPTIONS_ENV_PIECE_COLOUR: self.game_env_piece_colour, 'first_move': self.first_move}
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

    def _get_obs_for_env_trainer_model(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        # board_obs_state[0] == Player's piece board state,
        # board_obs_state[1] == Env's piece board state
        self.board_obs_state_for_env_trainer_model[0, :, :] = (self.board_state == self.game_env_piece_colour)
        self.board_obs_state_for_env_trainer_model[1, :, :] = (self.board_state == self.game_non_env_piece_colour)

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
                must_win_locations.append(trial_location)

        must_win_location_count = len(must_win_locations)
        ### non env makes the right move
        if action in must_win_locations:
            taken_right_move = True
            exist_must_win = True
            # logging.info(f'  taken must win, valid location : {action}')
            return exist_must_win, taken_right_move, must_win_location_count

        # logging.info(f' check must win, must win location: {must_win_locations}, len : {len(must_win_locations)}')
        # logging.info(f' sddd action in must_win: {not (action in must_win_locations)}')
        ### if there exists must-win location but non env player not take that move
        if (must_win_location_count > 0) and (not (action in must_win_locations)):
            exist_must_win = True
            taken_right_move = False
            return exist_must_win, taken_right_move, must_win_location_count

        return exist_must_win, taken_right_move, must_win_location_count

    def _non_env_move(self,action, game_step):
        valid_locations = GameEngine.get_valid_locations(self.board_state)
        is_non_env_game_win = False
        terminated = False
        is_draw = False
        reward = 0.0
        miss_must_win = False
        miss_must_defense = False
        non_env_valid_move = True

        if len(valid_locations) == 0 :  ### no possible move
            is_draw, reward, terminated = True, REWARD_DRAW , True  ## draw
        elif not GameEngine.is_valid_location(self.board_state, action):
            # logging.info(f'invalid move by non env , action: {action}')
            terminated , reward , non_env_valid_move= True, REWARD_NON_ENV_INVALID_MOVE , False ## invalid move
        else:
            ### check must-win col location
            exist_must_win, taken_right_move, must_win_location_count = self._check_must_win_location(action, self.game_non_env_piece_colour, valid_locations)
            # logging.info(f'exist must win : {exist_must_win},  taken_right_move : {taken_right_move}')
            if exist_must_win:
                terminated = True

                if taken_right_move:
                    is_non_env_game_win = GameEngine.drop_piece_from_top(self.board_state, action, self.game_non_env_piece_colour)
                    reward = REWARD_NON_ENV_WIN

                    if must_win_location_count > 1 : ## more than one winning location
                        reward += REWARD_MULTI_OPPORTUNITY_EXTRA * (must_win_location_count-1)

                    reward += (MAX_GAME_STEP - game_step) / (MAX_GAME_STEP - 4) * REWARD_WINNING_EARLY_EXTRA

                else:
                    miss_must_win = True
                    reward = REWARD_NON_ENV_MISSED_MUST_WIN

                    if must_win_location_count > 1:
                        reward -= REWARD_MULTI_OPPORTUNITY_EXTRA * (must_win_location_count-1)

                return terminated, reward, is_draw, is_non_env_game_win, miss_must_win, miss_must_defense, non_env_valid_move

            ###
            ### check must-defense col location, i.e. must-win of env_piece_colour
            exist_must_win, taken_right_move, must_win_location_count = self._check_must_win_location(action, self.game_env_piece_colour, valid_locations)
            if exist_must_win and not taken_right_move:
                reward = REWARD_NON_ENV_MISSED_MUST_DEFENSE - (REWARD_MULTI_OPPORTUNITY_EXTRA * must_win_location_count)
                reward -= (MAX_GAME_STEP - self.game_n_step) / (MAX_GAME_STEP - 4) * REWARD_LOSSING_EARLY_EXTRA

                terminated = True  ## does not take must-defense move
                miss_must_defense = True
                return terminated, reward, is_draw, is_non_env_game_win, miss_must_win, miss_must_defense, non_env_valid_move

            ### taken defensive move, continue
            GameEngine.drop_piece_from_top(self.board_state, action, self.game_non_env_piece_colour)

        return terminated, reward, is_draw, is_non_env_game_win, miss_must_win, miss_must_defense, non_env_valid_move

    def _env_move_decision(self, valid_locations):
        ### no matter which decision model, first check must take action
        ### check must-win col location
        for action in valid_locations:
            is_this_move_win = GameEngine.trial_drop_piece_from_top(self.board_state, action, self.game_env_piece_colour)
            if is_this_move_win:
                return action

        ### check must-defense col location
        for action in valid_locations:
            is_this_move_win = GameEngine.trial_drop_piece_from_top(self.board_state, action,
                                                                    self.game_non_env_piece_colour)
            if is_this_move_win:
                return action

        if self.env_trainer_model:
            random_number = random.random()
            if random_number < EXPLORATION_RATE:
                ## use random
                env_action_col = random.choice(valid_locations)
            else:
                obs = self._get_obs_for_env_trainer_model()
                env_action_col, _states = self.env_trainer_model.predict(obs)
                ### env trainer model can output invalid action, check
                if not env_action_col in valid_locations:
                    env_action_col = random.choice(valid_locations)
                # logging.info(f'env trainer model: action : {env_action_col}')
        else:
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
        is_env_valid_move = True
        is_non_env_game_win = False
        is_non_env_valid_move = False
        miss_must_win = False
        miss_must_defense = False
        is_draw = False

        logging.info(f'---- new step ----- , step number : {self.game_n_step}')
        logging.info(f'non env action : {action}')


        ## non env move
        terminated, reward, is_draw, is_non_env_game_win, miss_must_win, miss_must_defense, is_non_env_valid_move = self._non_env_move(action, self.game_n_step)

        if not terminated:
            ###
            ### response by env. env's turn to move
            ###
            valid_locations = GameEngine.get_valid_locations(self.board_state)
            if len(valid_locations) == 0:  ### no possible move, draw
                is_draw , reward , terminated = True, REWARD_DRAW, True
            else:
                ### env's to pick a move
                env_action_col = self._env_move_decision(valid_locations)
                logging.info(f'env action : {env_action_col}')
                is_env_game_win = GameEngine.drop_piece_from_top(self.board_state, env_action_col, self.game_env_piece_colour)

                if is_env_game_win:
                    reward = REWARD_NON_ENV_LOSS
                    ### additional -ve reward if loss too early
                    reward -= (MAX_GAME_STEP - self.game_n_step) / (MAX_GAME_STEP - 4 ) * REWARD_LOSSING_EARLY_EXTRA

                    terminated = True

                ## new valid location after env response
                valid_locations = GameEngine.get_valid_locations(self.board_state)
                if len(valid_locations) == 0:  ### no possible move, draw
                    is_draw, reward, terminated = True, REWARD_DRAW , True

        observation = self._get_obs()
        info = { 'step' : self.game_n_step, 'env_win': is_env_game_win, 'non_env_win' : is_non_env_game_win,
                 'env_valid_move' : is_env_valid_move, 'non_env_valid_move' : is_non_env_valid_move , 'is_draw' : is_draw ,
                 'miss_must_win' : miss_must_win , 'miss_must_defense': miss_must_defense ,
                 'first_move': self.first_move}

        # logging.info(f' observation : \n{observation}')
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        Renderer.render(self.board_state, mode)
