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
MODEL_NAME_ENV_TRAINER = "connect_four_env_trainer"

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


def training_phase(env_trainer_model):
    # Set the global logging level to INFO
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    env = ConnectFourEnv.ConnectFourEnv()
    env.reset(options= { ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL:  env_trainer_model})

    model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [128, 128, 128]})

    logging.info(f'model: {model.policy}')
    # model.learn(total_timesteps=25000)
    # model.learn(total_timesteps=50000)
    model.learn(total_timesteps=1000)
    model.save(MODEL_NAME_TRAINED)

    del model  # remove to demonstrate saving and loading

    rootLogger.setLevel(logging.INFO)
    logging.info(f"training finished")

    return env

def inference_phase(env, env_trainer_model):
    model = A2C.load(MODEL_NAME_TRAINED)

    obs , info = env.reset(options= { ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL:  env_trainer_model})
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        action, _states = model.predict(obs)
        observation, reward, terminated, truncated, info = env.step(action)

        logging.info(f'step finish. reward : {reward}, action: {action} , info: {info}')
        env.render("human")

def prepare_env_trainer_model():

    logging.info(f'loading env trainer model')
    env_trainer_model = A2C.load(MODEL_NAME_ENV_TRAINER)
    return env_trainer_model

def reinforcement_main():
    # environments

    env_trainer_model = prepare_env_trainer_model()

    env = training_phase(env_trainer_model)
    inference_phase(env, env_trainer_model)

    logging.info(f'end')

if __name__ == "__main__":
    setupLogging()

    init_system()

    # supervised_main()
    reinforcement_main()
