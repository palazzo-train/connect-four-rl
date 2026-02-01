import os
import sys
import logging
from game_env import ConnectFourEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device


MODEL_NAME_TRAINED = "connect_four"


## quick smoke test
# PARAMETER_EVAL_RUN_COUNT = 100
# PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 1000
# PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 1

## quick train
PARAMETER_EVAL_RUN_COUNT = 100
PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 5000
PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 2

# PARAMETER_EVAL_RUN_COUNT = 1000
# PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 50000
# PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 10


def init_system():
    logging.info(f'torch version {torch.version.cuda}')
    logging.info(f'torch cuda available: {torch.cuda.is_available()}')
    #
    # This will automatically use 'cuda' if a GPU is available
    # device = get_device("auto")
    device = get_device("cpu")
    logging.info(f"Current device: {device}")


def setupLogging():
    fileName = 'app.log'
    logPath = '.'
    path = os.path.join( logPath , fileName )

    format='%(asctime)s %(levelname)-8s - %(message)s'
    logFormatter = logging.Formatter(format)
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)


def training_phase():
    # rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.WARNING)

    for i in range(PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION):
        # model_name = f'{MODEL_NAME_TRAINED}_iter_{str(i)}'
        model_name = MODEL_NAME_TRAINED
        logging.info(f'training iteration {i}. loading env trainer model [{model_name}]')
        env_trainer_model = A2C.load(model_name)

        env = ConnectFourEnv.ConnectFourEnv()
        env.reset(options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_trainer_model})
        agent_model = A2C.load(model_name, env=env)


        # new_model_name = f'{MODEL_NAME_TRAINED}_iter_{str(i+ 1)}'
        new_model_name = MODEL_NAME_TRAINED

        training_iteration(agent_model, env_trainer_model, i, new_model_name)
        del env_trainer_model
        del agent_model
        logging.info(f'saved new model [{new_model_name}]')

    return new_model_name


def training_iteration(agent_model, env_trainer_model, training_iter, new_model_name):

    # Set the global logging level to INFO
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    # model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [128, 128, 128]})
    model = agent_model

    logging.info(f'model: {model.policy}')
    # model.learn(total_timesteps=25000)
    # model.learn(total_timesteps=50000)
    model.learn(total_timesteps= PARAMETER_MODEL_TRAINING_ITERATION_COUNT)
    model.save(new_model_name)

    del model  # remove to demonstrate saving and loading

    rootLogger.setLevel(logging.INFO)
    logging.info(f"training finished")


def evaluation_phase(env, env_trainer_model):
    logging.info(f'evaluation phase')
    model = A2C.load(MODEL_NAME_TRAINED)

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    win_count = 0
    loss_count = 0
    draw_count = 0

    win_average_steps = 0
    win_smallest_steps = 100
    loss_average_steps = 0
    loss_smallest_steps = 100
    draw_average_steps = 0
    invalid_move_count = 0
    miss_must_win_count = 0
    miss_must_defense_count = 0

    def running_average(step, prev_average, prev_count):
        new_count = prev_count + 1
        if prev_average == 0 :
           new_average = step
        else:
            new_average = prev_average * prev_count / new_count + step / new_count

        return new_count, new_average

    eval_game_count = PARAMETER_EVAL_RUN_COUNT
    report_interval = eval_game_count / 10
    report_interval_count = 0
    for i in range(eval_game_count):
        obs, info = env.reset(options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_trainer_model})
        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            action, _states = model.predict(obs)
            observation, reward, terminated, truncated, info = env.step(action)

        if report_interval_count > report_interval:
            report_interval_count = 0
            rootLogger.setLevel(logging.INFO)
            logging.info(f'evaluation progress {i/eval_game_count * 100}%')
            rootLogger.setLevel(logging.WARNING)
        else:
            report_interval_count += 1

        step = info['step']
        if info['non_env_win']:
            win_count, win_average_steps = running_average(step, win_average_steps, win_count)
            if step < win_smallest_steps :
                win_smallest_steps = step

        if info['env_win']:
            loss_count, loss_average_steps = running_average(step, loss_average_steps, loss_count)
            if step < loss_smallest_steps :
                loss_smallest_steps = step

        if info['is_draw']:
            draw_count, draw_average_steps = running_average(step, draw_average_steps, draw_count)

        if not info['non_env_valid_move'] :
            invalid_move_count = invalid_move_count + 1

        if info['miss_must_win']:
            miss_must_win_count = miss_must_win_count + 1

        if info['miss_must_defense']:
            miss_must_defense_count = miss_must_defense_count + 1

    rootLogger.setLevel(logging.INFO)
    eval_info = {
            'win_count' : win_count ,
            'loss_count' : loss_count ,
            'draw_count' : draw_count ,
            'win_average_steps' : win_average_steps ,
            'win_smallest_steps' : win_smallest_steps  ,
            'loss_average_steps' : loss_average_steps ,
            'loss_smallest_steps' : loss_smallest_steps  ,
            'draw_average_steps' : draw_average_steps ,
            'invalid_move_count' : invalid_move_count ,
            'miss_must_win_count' : miss_must_win_count ,
            'miss_must_defense_count' : miss_must_defense_count }

    return eval_info


def demo_phase(env, env_trainer_model):
    model = A2C.load(MODEL_NAME_TRAINED)

    obs , info = env.reset(options= { ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL:  env_trainer_model})
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        action, _states = model.predict(obs)
        observation, reward, terminated, truncated, info = env.step(action)

        logging.info(f'step finish. reward : {reward}, action: {action} , info: {info}')
        env.render("human")


def reinforcement_main():
    # environments

    new_model_name = training_phase()

    env = ConnectFourEnv.ConnectFourEnv()
    env_trainer_model = A2C.load(new_model_name)
    env.reset(options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_trainer_model})

    eval_info = evaluation_phase(env, env_trainer_model)
    logging.info(f'eval info: {eval_info}')
    demo_phase(env, env_trainer_model)

    logging.info(f'end')

if __name__ == "__main__":
    setupLogging()

    init_system()

    # supervised_main()
    reinforcement_main()
