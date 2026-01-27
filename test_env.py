import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

print(f'torch version {torch.version.cuda}')
print(f'torch {torch.cuda.is_available()}')

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.utils import get_device
#
# This will automatically use 'cuda' if a GPU is available
device = get_device("auto")
print(f"Current device: {device}")

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
