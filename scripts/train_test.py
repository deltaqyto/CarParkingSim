import os
import time
import numpy as np
import torch
import inspect

# Disable matplotlib GUI
import matplotlib

matplotlib.use('Agg')

# Stable Baselines3 imports
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import simulation wrapper
from Simulation.simulation_wrapper import SimulationWrapper
from Simulation.environments import get_basic_env


class BenchmarkCallback(BaseCallback):
    """Callback for tracking performance metrics and stopping after 2 minute."""

    def __init__(self, verbose=0):
        super(BenchmarkCallback, self).__init__(verbose)
        # Initialize timers
        self.start_time = None
        self.end_time = None
        self.total_iterations = 0
        self.timings = {
            'load': 0,
            'env_time': 0,
            'model_inference_time': 0,
            'model_train_time': 0
        }
        self.current_section = None
        self.section_start_time = None

    def _on_training_start(self):
        """Called when training starts."""
        self.start_time = time.time()
        self.end_time = self.start_time + 120  # 2 minute benchmark

    def _on_step(self):
        """Called after each step in the environment."""
        self.total_iterations += 1
        # Stop after 1 minute
        if time.time() >= self.end_time:
            return False
        return True

    def start_timing(self, section):
        """Start timing a specific section."""
        self.section_start_time = time.time()
        self.current_section = section

    def end_timing(self):
        """End timing the current section."""
        if self.current_section:
            elapsed = time.time() - self.section_start_time
            self.timings[self.current_section] += elapsed
            self.current_section = None


def run_benchmark(num_envs=14):
    """Run a 2-minute benchmark of the TD3 reinforcement learning algorithm."""
    # Create callback for tracking metrics
    callback = BenchmarkCallback()

    # Start timing the load phase
    callback.start_timing('load')

    # Parameters for the benchmark
    params = {
        'num_envs': min(num_envs, max(1, os.cpu_count())),
        'action_dim': 2,  # throttle, steering
        'batch_size': 256,
        'seed': 41,
        'exploration_noise': 0.1,
        'start_timesteps': 1000,  # Reduced for benchmark
        'buffer_size': 1_000_000,
        'learning_rate': 3e-4,
        'net_size': [400, 300],
        'total_timesteps': 3_000_000,  # Will be cut off at 1 minute
    }

    # Set seeds for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # Get the environment
    env_fn = get_basic_env(False, 2, 1.57)

    # Create vectorized environment
    env = SubprocVecEnv([
        lambda: SimulationWrapper(env_fn, i, params['seed'])
        for i in range(params['num_envs'])
    ])

    # Create action noise for exploration
    n_actions = params['action_dim']
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=params['exploration_noise'] * np.ones(n_actions)
    )

    # Create model
    model = TD3(
        "MlpPolicy",
        env=env,
        learning_rate=params['learning_rate'],
        buffer_size=params['buffer_size'],
        learning_starts=params['start_timesteps'],
        batch_size=params['batch_size'],
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=params['net_size']),
        verbose=0,  # Disable verbose logging
        seed=params['seed']
    )

    # End load timing
    callback.end_timing()

    # Get the original _sample_action method signature to understand its parameters
    original_sample_action = model._sample_action
    original_train = model.train
    original_step = env.step

    # Patch methods to add timing
    def patched_sample_action(*args, **kwargs):
        callback.start_timing('model_inference_time')
        result = original_sample_action(*args, **kwargs)
        callback.end_timing()
        return result

    def patched_train(*args, **kwargs):
        callback.start_timing('model_train_time')
        result = original_train(*args, **kwargs)
        callback.end_timing()
        return result

    def patched_step(*args, **kwargs):
        callback.start_timing('env_time')
        result = original_step(*args, **kwargs)
        callback.end_timing()
        return result

    # Apply patches more carefully with proper binding
    model._sample_action = patched_sample_action
    model.train = patched_train
    env.step = patched_step

    # Initialize metrics dictionary to store results
    metrics = {
        'num_envs': num_envs,
        'fps': 0,
        'steps': 0,
        'load_time': 0,
        'env_time': 0,
        'inference_time': 0,
        'train_time': 0,
        'mspt': 0,
        'total_env_steps': 0  # Added to track total environment steps
    }

    try:
        print("Starting benchmark (will run for 2 minute)...")

        # Run training with timing
        model.learn(
            total_timesteps=params['total_timesteps'],  # Will be cut short by callback
            callback=callback,
            log_interval=100000  # Essentially disable logging
        )
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Restore original methods
        model._sample_action = original_sample_action
        model.train = original_train
        env.step = original_step
        env.close()

    # Calculate benchmark results
    total_time = time.time() - callback.start_time if callback.start_time else 0
    fps = callback.total_iterations / (total_time - callback.timings['load']) if total_time > callback.timings['load'] else 0

    # Calculate total environment steps (iterations * num_envs)
    total_env_steps = callback.total_iterations * num_envs

    # Store metrics
    metrics['fps'] = fps
    metrics['steps'] = callback.total_iterations
    metrics['load_time'] = callback.timings['load']
    metrics['env_time'] = callback.timings['env_time']
    metrics['inference_time'] = callback.timings['model_inference_time']
    metrics['train_time'] = callback.timings['model_train_time']
    metrics['total_env_steps'] = total_env_steps  # Store total environment steps

    # Calculate MSPT (milliseconds per timestep)
    if callback.total_iterations > 0:
        metrics['mspt'] = (callback.timings['env_time'] * 1000) / callback.total_iterations

    # Calculate MSPE (milliseconds per environment step) - a more meaningful metric for comparison
    if total_env_steps > 0:
        metrics['mspe'] = (callback.timings['env_time'] * 1000) / total_env_steps
    else:
        metrics['mspe'] = 0

    # Print benchmark results
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total iterations: {callback.total_iterations}")
    print(f"Total environment steps: {total_env_steps}")
    print("\nTime breakdown:")
    print(f"  Load time: {callback.timings['load']:.2f} seconds")
    print(f"  Environment time: {callback.timings['env_time']:.2f} seconds")
    print(f"  Model inference time: {callback.timings['model_inference_time']:.2f} seconds")
    print(f"  Model train time: {callback.timings['model_train_time']:.2f} seconds")
    print(f"\nFPS estimate (ignoring initial load time): {fps:.2f}")
    print(f"MSPT (per iteration): {metrics['mspt']:.2f} ms")
    print(f"MSPE (per environment step): {metrics['mspe']:.2f} ms")
    print("=============================\n")

    # Return the metrics dictionary
    return metrics
