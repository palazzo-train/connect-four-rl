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


def reforcement_main_test():

    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_cartpole")

    del model  # remove to demonstrate saving and loading

    logging.info(f"training finished")
    model = A2C.load("a2c_cartpole")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

def reforcement_main():
    # reforcement_main_test()

    # environments
    env = ConnectFourEnv()

    # Set the global logging level to INFO
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    model = A2C("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=25000)
    model.learn(total_timesteps=1000)
    model.save("connect_four")

    del model  # remove to demonstrate saving and loading


    rootLogger.setLevel(logging.INFO)
    logging.info(f"training finished")

    model = A2C.load("connect_four")

    obs , info = env.reset()
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        action, _states = model.predict(obs)
        observation, reward, terminated, truncated, info = env.step(action)

        logging.info(f'step finish. reward : {reward}, action: {action} , info: {info}')
        env.render("human")

    logging.info(f'end')

if __name__ == "__main__":
    setupLogging()

    init_system()

    # supervised_main()
    reforcement_main()
