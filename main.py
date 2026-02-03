import os
import sys
import logging
from datetime import datetime
import argparse
from game_env import ConnectFourEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device


MODEL_NAME_TRAINED = "connect_four"
MODEL_NAME_EVALUATOR = "connect_four"

## quick train
PARAMETER_EVAL_RUN_COUNT = 100
PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 5000
PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 2

TENSORBOARD_BASE= 'connect_4_tensorboard'


def parameter_update(speed):
    global PARAMETER_EVAL_RUN_COUNT
    global PARAMETER_MODEL_TRAINING_ITERATION_COUNT
    global PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION

    logging.info(f'parameter update. speed: {speed} , type : {type(speed)}')

    if speed == 0 :
        PARAMETER_EVAL_RUN_COUNT = 100
        PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 1000
        PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 1
    elif speed == 1 :
        PARAMETER_EVAL_RUN_COUNT = 100
        PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 500
        PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 4
    elif speed == 2:
        PARAMETER_EVAL_RUN_COUNT = 1000
        PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 20000
        PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 4
    elif speed == 3:
        PARAMETER_EVAL_RUN_COUNT = 1000
        PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 50000
        PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 4
    elif speed == 4:
        PARAMETER_EVAL_RUN_COUNT = 1000
        PARAMETER_MODEL_TRAINING_ITERATION_COUNT = 50000
        PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION = 10

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


def training_phase(tensorboard_log, reset_num_timesteps):

    # rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.WARNING)

    for i in range(PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION):
        # model_name = f'{MODEL_NAME_TRAINED}_iter_{str(i)}'
        model_name = MODEL_NAME_TRAINED
        logging.info(f'training iteration {i} of {PARAMETER_MODEL_TRAINING_MODEL_SWAP_ITERATION}. loading env trainer model [{model_name}]')
        # env_trainer_model = A2C.load(model_name, verbose=1,tensorboard_log="./connect_4_tensorboard")
        env_trainer_model = A2C.load(model_name)

        env = ConnectFourEnv.ConnectFourEnv( options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_trainer_model})
        env.reset()

        agent_model = A2C.load(model_name, env=env, verbose=1,tensorboard_log=tensorboard_log)
        new_model_name = MODEL_NAME_TRAINED

        training_iteration(agent_model, env_trainer_model, i, new_model_name, reset_num_timesteps)


        del env_trainer_model
        del agent_model
        logging.info(f'saved new model [{new_model_name}]')
        reset_num_timesteps = False

    return new_model_name


def create_model():
    import torch as th
    import torch.nn as nn
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    class CustomCNN(BaseFeaturesExtractor):
        """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

        def __init__(self, observation_space: spaces.Box, features_dim: int = 256, net_arch: list[int] = [32 , 64, 96]):
            super().__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            # Re-ordering will be done by pre-preprocessing or wrapper
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, net_arch[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(net_arch[0], net_arch[1], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(net_arch[1], net_arch[2], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(
                    th.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))



    model_name = MODEL_NAME_TRAINED
    logging.info(f'create model : {model_name}')
    env = ConnectFourEnv.ConnectFourEnv()
    # model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [128, 128, 128]})
    policy_kwargs = dict(
        net_arch= dict(vf=[64], pi=[128]),
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128, net_arch=[64, 64, 96]),
    )
    model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    policy = model.policy
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    model.save(model_name)
    logging.info(f'model policy: {model.policy}')
    logging.info(f'model number of trainable parameters: {total_params}')
    logging.info(f'model : {model_name} saved')

def training_iteration(agent_model, env_trainer_model, training_iter, new_model_name, reset_num_timesteps):

    # Set the global logging level to INFO
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    # model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [128, 128, 128]})
    model = agent_model

    logging.info(f'model: {model.policy}')
    # model.learn(total_timesteps=25000)
    # model.learn(total_timesteps=50000)
    model.learn(total_timesteps= PARAMETER_MODEL_TRAINING_ITERATION_COUNT, reset_num_timesteps=reset_num_timesteps)
    model.save(new_model_name)

    # del model  # remove to demonstrate saving and loading

    rootLogger.setLevel(logging.INFO)
    logging.info(f"training iteration {training_iter} finished.")


def evaluation_phase(tensorboard_log):
    from torch.utils.tensorboard import SummaryWriter

    logging.info(f'evaluation phase')
    tensorboard_eval_log = f"{tensorboard_log}/eval_game/"

    model = A2C.load(MODEL_NAME_TRAINED)
    evaluator_model_name = MODEL_NAME_EVALUATOR

    env_evaluator_model = A2C.load(evaluator_model_name)
    eval_env = ConnectFourEnv.ConnectFourEnv(options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_evaluator_model})
    eval_env.reset()

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
        obs, info = eval_env.reset()
        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            action, _states = model.predict(obs)
            observation, reward, terminated, truncated, info = eval_env.step(action)

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

    writer = SummaryWriter(tensorboard_eval_log)
    for d in eval_info:
        writer.add_scalar(f'eval_game/{d}', eval_info[d], model.num_timesteps)

    return eval_info


def demo_phase(env, env_trainer_model):
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
    # Get the current date and time as a datetime object
    now = datetime.now()
    # Format the datetime object into a string (e.g., YYYY-MM-DD HH:MM:SS)
    timelabel = now.strftime("%Y-%m-%d_%H.%M.%S")
    tensorboard_log = f"./{TENSORBOARD_BASE}/{timelabel}"
    reset_num_timesteps = True

    for i in range(10):
        learning_and_eval_loop(tensorboard_log, reset_num_timesteps)
        reset_num_timesteps = False

def learning_and_eval_loop(tensorboard_log, reset_num_timesteps):
    new_model_name = training_phase(tensorboard_log, reset_num_timesteps)

    env_trainer_model = A2C.load(new_model_name)
    env = ConnectFourEnv.ConnectFourEnv(options={ConnectFourEnv.OPTIONS_ENV_TRAINER_MODEL: env_trainer_model})
    env.reset()

    eval_info = evaluation_phase(tensorboard_log)

    logging.info(f'eval info: {eval_info}')
    demo_phase(env, env_trainer_model)

    logging.info(f'end')


if __name__ == "__main__":
    setupLogging()

    parser = argparse.ArgumentParser(description='Connect Four.')
    parser.add_argument("--mode", default='train', help='train, test, init')
    parser.add_argument("--speed" , help='0,1,2,3' , default=0, type=int)
    args = parser.parse_args()

    parameter_update(args.speed)
    logging.info(f'arguments: {args}')
    init_system()


    if args.mode == 'train':
        reinforcement_main()
    elif args.mode == 'init':
        create_model()
