import os
import sys
import logging
from game_env.ConnectFourEnv import ConnectFourEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device


MODEL_NAME_TRAINED = "connect_four"


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


def training_phase(env):
    # Set the global logging level to INFO
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [128, 128, 128]})

    logging.info(f'model: {model.policy}')
    # model.learn(total_timesteps=25000)
    model.learn(total_timesteps=50000)
    # model.learn(total_timesteps=1000)
    model.save(MODEL_NAME_TRAINED)

    del model  # remove to demonstrate saving and loading

    rootLogger.setLevel(logging.INFO)
    logging.info(f"training finished")

def inference_phase(env):
    model = A2C.load(MODEL_NAME_TRAINED)

    obs , info = env.reset()
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        action, _states = model.predict(obs)
        observation, reward, terminated, truncated, info = env.step(action)

        logging.info(f'step finish. reward : {reward}, action: {action} , info: {info}')
        env.render("human")

def reinforcement_main():
    # environments
    env = ConnectFourEnv()

    training_phase(env)
    inference_phase(env)

    logging.info(f'end')

if __name__ == "__main__":
    setupLogging()

    init_system()

    # supervised_main()
    reinforcement_main()
