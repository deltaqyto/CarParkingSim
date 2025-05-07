import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from multiprocessing import cpu_count

from Simulation.environments import get_basic_env
from AI.single_learning import do_single_learning


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)

        base_params = {
            'num_envs': min(14, max(1, cpu_count())),
            'action_dim': 2,  # throttle, steering
            'batch_size': 256,
            'total_timesteps': 3_000_000,
            'save_freq': 20000,
            'eval_episodes': 10,
            'seed': 41,
            'exploration_noise': 0.1,
            'start_timesteps': 25000,  # Random exploration steps
            'buffer_size': 1_000_000,
            'learning_rate': 3e-4,
            'net_size': [400, 300],
        }

        do_single_learning(get_basic_env(False, 2, 1.57), base_params)  # Start training with the basic environment and some hyperparameters
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        plt.close('all')
