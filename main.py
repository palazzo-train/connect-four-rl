import os
import sys
import logging



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


def system_init():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def reforcement_main():
    import torch

    logging.info(f'torch version {torch.version.cuda}')
    logging.info(f'torch cuda available: {torch.cuda.is_available()}')

    import gymnasium as gym
    from stable_baselines3 import A2C
    from stable_baselines3.common.env_util import make_vec_env

    from stable_baselines3.common.utils import get_device
    #
    # This will automatically use 'cuda' if a GPU is available
    device = get_device("auto")
    # device = get_device("cpu")
    logging.info(f"Current device: {device}")

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


if __name__ == "__main__":
    system_init()
    setupLogging()

    # supervised_main()
    reforcement_main()
