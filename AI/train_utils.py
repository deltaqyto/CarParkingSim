import os

import time
import numpy as np
import glob
import threading
import torch


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stable Baselines3 imports
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Simulation.simulation_wrapper import SimulationWrapper


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True


def evaluate_model(environment, model_path, num_episodes=10):
    """Evaluate a given model by running episodes and collecting statistics."""
    env = environment()

    try:
        model = TD3.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    # Statistics
    episode_rewards = []
    episode_steps = []
    goal_distances = []
    stop_reasons_dict = {}
    reward_breakdown_dict = {}

    # Run episodes
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Get clean action from model
            action, _ = model.predict(observation, deterministic=True)

            # Step environment
            observation, reward, done, _, state = env.step(action)

            episode_reward += reward
            steps += 1

            assert type(state) is not None  # Env didn't provide a proper state dict, or it never ran

            for reason in state['stop_reasons']:  # Sum stop reasons separately
                stop_reasons_dict[reason] = 1 + stop_reasons_dict.get(reason, 0)
            for name, reward in state['reward_types'].items():  # Sum episode rewards separately
                reward_breakdown_dict[name] = reward + reward_breakdown_dict.get(name, 0)
            goal_distances.append(state['closest_goal']['distance'])

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

    # Calculate averages
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    avg_goal_distance = np.mean(goal_distances)

    for name, value in reward_breakdown_dict.items():
        reward_breakdown_dict[name] = value / num_episodes
    for name, value in stop_reasons_dict.items():
        stop_reasons_dict[name] = value / num_episodes

    return {
        'timestep': get_timestep_from_path(model_path),
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'stop_reasons': stop_reasons_dict,
        'reward_breakdown': reward_breakdown_dict,
        'average_goal_distance': avg_goal_distance,
    }


def get_timestep_from_path(model_path):
    """Extract timesteps from model path if possible."""
    try:
        # For paths like: models/td3_ABC1/ABC1_50000_steps.zip
        filename = os.path.basename(model_path)
        parts = filename.split('_')
        if len(parts) >= 2 and parts[-1] == "steps.zip":
            return int(parts[-2])
        elif len(parts) >= 2 and parts[-1] == "steps":
            return int(parts[-2])
        elif "final" in filename:
            return float('inf')  # Final model
        return -1  # Unknown timesteps
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print(f"Got exception: \n{e}\nContinuing...")
        return -1


def find_td3_models(search_path, model_code=None):
    """Find TD3 models matching the given code."""
    if model_code is None:
        return []

    model_dir = os.path.join(search_path, f"td3_{model_code}")
    if not os.path.exists(model_dir):
        return []

    # Look for model files with .zip extension
    model_files = glob.glob(os.path.join(model_dir, "*.zip"))
    return model_files


def get_best_model(model_paths):
    """Get the best model from a list of model paths."""
    best_model_path = None
    best_model = 0
    for model_path in model_paths:
        timesteps = get_timestep_from_path(model_path)
        if timesteps > best_model:
            best_model_path = model_path
            best_model = timesteps

    return best_model_path


class ModelEvaluationMonitor:
    """Monitor for new model checkpoints and evaluate them automatically."""

    def __init__(self, environment, model_dir, train_id, num_episodes=10):
        self.environment = environment
        self.model_dir = model_dir
        self.train_id = train_id
        self.num_episodes = num_episodes
        self.known_models = set()
        self.results = []
        self.running = True
        self.eval_thread = None

        # Paths for saving results
        self.results_path = os.path.join(model_dir, f"{train_id}_eval_results.txt")
        self.plot_path = os.path.join(model_dir, f"{train_id}_eval_plot.png")

        # Initialize results file with header
        with open(self.results_path, 'w') as f:
            f.write(f"Evaluation Results for {train_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestep, Average reward, Avg Steps, Stop reasons, Rewards, Goal distances\n")
            f.write("-" * 80 + "\n")

    def start(self):
        """Start the monitoring thread."""
        self.eval_thread = threading.Thread(target=self._monitor_loop)
        self.eval_thread.daemon = True
        self.eval_thread.start()
        print(f"Started evaluation monitor for model {self.train_id}")

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.eval_thread:
            self.eval_thread.join(timeout=2.0)
        print("Evaluation monitor stopped")

    def _monitor_loop(self):
        while self.running:
            try:
                # Look for new model files
                pattern = os.path.join(self.model_dir, f"{self.train_id}_*_steps.zip")
                model_files = glob.glob(pattern)

                # Check for final model
                final_pattern = os.path.join(self.model_dir, f"{self.train_id}_final.zip")
                final_models = glob.glob(final_pattern)
                model_files.extend(final_models)

                # Filter for new models
                new_models = [m for m in model_files if m not in self.known_models]

                for model_path in sorted(new_models, key=get_timestep_from_path):
                    # Mark as known before evaluation to prevent duplicate evaluations
                    self.known_models.add(model_path)

                    # Evaluate the model
                    print(f"Evaluating model: {os.path.basename(model_path)}")
                    result = evaluate_model(
                        environment=self.environment,
                        model_path=model_path,
                        num_episodes=self.num_episodes,
                    )

                    if result:
                        self._append_result(result)
                        self._update_plot(result)

                # Sleep before next check
                time.sleep(1.0)
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(5.0)  # Wait longer on error

    def _append_result(self, result):
        """Append a single result to the results file."""
        try:
            with open(self.results_path, 'a') as f:
                f.write(f"{result}\n")
        except Exception as e:
            print(f"Error writing result: {e}")

    def _update_plot(self, result):
        """Update the evaluation results plot"""
        self.results.append(result)
        try:
            # Filter out 'final' model for plotting by timestep
            plot_results = [r for r in self.results if r['timestep'] != float('inf')]
            if len(plot_results) < 1:
                plt.figure(figsize=(1, 1))
                plt.tight_layout()
                plt.savefig(self.plot_path)
                plt.close()
                return  # Need at least 1 point to make a plot
            # Sort by timestep
            plot_results.sort(key=lambda x: x['timestep'])
            # Extract data for plotting
            timestep = [r['timestep'] for r in plot_results]
            avg_rewards = [r['avg_reward'] for r in plot_results]
            avg_steps = [r['avg_steps'] for r in plot_results]
            stop_reasons = [r['stop_reasons'] for r in plot_results]
            reward_breakdowns = [r['reward_breakdown'] for r in plot_results]
            goal_distances = [r['average_goal_distance'] for r in plot_results]

            # Create figure with 2x2 grid layout
            plt.figure(figsize=(20, 10))

            # Plot 1: Stop Reasons (top-left)
            plt.subplot(2, 2, 1)
            # Get all unique stop reason keys across all dictionaries
            all_stop_reasons = set()
            for sr in stop_reasons:
                all_stop_reasons.update(sr.keys())
            # Plot each stop reason as its own line
            for reason in all_stop_reasons:
                reason_values = []
                for i, sr in enumerate(stop_reasons):
                    value = sr.get(reason, 0)  # Default to 0 if key is missing
                    reason_values.append((timestep[i], value))
                if reason_values:
                    x_vals, y_vals = zip(*reason_values)
                    plt.plot(x_vals, y_vals, 'o-', label=reason)
            plt.xlabel('Training Timestep')
            plt.ylabel('Rate')
            plt.title('Stop Reasons per Timestep')
            plt.legend()
            plt.grid(True)

            # Plot 2: Average Steps & Rewards with dual y-axes (top-right)
            plt.subplot(2, 2, 2)
            fig = plt.gcf()
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create a second y-axis
            # Plot average steps on left y-axis
            ax1.plot(timestep, avg_steps, 'b-', label='Avg Steps')
            ax1.set_xlabel('Training Timestep')
            ax1.set_ylabel('Steps', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            # Plot average rewards on right y-axis
            ax2.plot(timestep, avg_rewards, 'r-', label='Avg Rewards')
            ax2.set_ylabel('Reward', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            plt.title('Average Steps and Rewards per Episode')
            plt.grid(True)

            # Plot 3: Goal distances (bottom-left)
            plt.subplot(2, 2, 3)
            plt.plot(timestep, goal_distances, 'mo-')
            plt.xlabel('Training Timestep')
            plt.ylabel('Goal Distance')
            plt.title('Average Distance to Goal per Episode')
            plt.grid(True)

            # Plot 4: Reward breakdowns (bottom-right)
            plt.subplot(2, 2, 4)
            # Get all unique reward breakdown keys across all dictionaries
            all_breakdown_keys = set()
            for rb in reward_breakdowns:
                all_breakdown_keys.update(rb.keys())
            # Plot each reward component as its own line with different colors
            for key in all_breakdown_keys:
                # Extract values for this component, handling missing keys
                component_values = []
                for i, rb in enumerate(reward_breakdowns):
                    if key in rb:
                        component_values.append((timestep[i], rb[key]))
                if component_values:
                    x_vals, y_vals = zip(*component_values)
                    plt.plot(x_vals, y_vals, '-', label=key)
            plt.xlabel('Training Timestep')
            plt.ylabel('Reward')
            plt.title('Reward Breakdown per Episode')
            plt.legend(loc='upper left')
            plt.grid(True)

            plt.suptitle(f'Model Evaluation Results - {self.train_id}', fontsize=16)
            plt.tight_layout()
            # Save figure without displaying
            plt.savefig(self.plot_path)
            plt.close()  # Important! Close the figure to avoid memory leaks
            print(f"Updated evaluation plot saved to {self.plot_path}")
        except Exception as e:
            print(f"Error updating plot: {e}")


def setup_model_training(environment, params, train_id, model_path=None, search_path="models"):
    """Common setup for both single and curriculum learning.

    Args:
        environment: Environment factory function
        params: Dictionary of training parameters
        train_id: Training ID string
        model_path: Optional path to model to load for continued training
        search_path: Base directory for models

    Returns:
        Tuple of (model, env, model_dir, monitor)
    """
    # Create result directories
    model_dir = os.path.join(search_path, f"td3_{train_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Save environment digest
    try:
        env_string = environment().get_digest()
        param_string = str(params)
        digest = env_string + '\n\nparams(' + param_string + ")"
        digest_path = os.path.join(model_dir, "digest.txt")
        with open(digest_path, "w") as f:
            f.write(digest)
        print(f"Environment digest saved to: {digest_path}")
    except Exception as e:
        print(f"Failed to save environment digest: {e}")

    # Set seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # Create vectorized environment
    env = SubprocVecEnv([lambda: SimulationWrapper(environment, i, params['seed'])
                         for i in range(params['num_envs'])])

    # Define action noise for exploration
    n_actions = params['action_dim']
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=params['exploration_noise'] * np.ones(n_actions)
    )

    if model_path:
        # Load existing model
        print(f"Loading model from {model_path}...")
        model = TD3.load(
            model_path,
            env=env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            batch_size=params['batch_size'],
            action_noise=action_noise,
        )
        print("Model loaded successfully!")
    else:
        # Create new model
        model = TD3(
            "MlpPolicy",
            env=env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            learning_starts=params['start_timesteps'],
            batch_size=params['batch_size'],
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=params['net_size']),
            verbose=1,
            seed=params['seed']
        )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=params['save_freq'] // params['num_envs'],
        save_path=model_dir,
        name_prefix=train_id,
        verbose=1
    )

    # Start evaluation monitor
    monitor = ModelEvaluationMonitor(
        environment=lambda: SimulationWrapper(environment, 0, params['seed']),
        model_dir=model_dir,
        train_id=train_id,
        num_episodes=params['eval_episodes']
    )
    monitor.start()

    return model, env, checkpoint_callback, model_dir, monitor


def train_model(model, env, checkpoint_callback, model_dir, train_id, total_timesteps, monitor=None):
    """Common training function for both single and curriculum learning.

    Args:
        model: TD3 model
        env: Training environment
        checkpoint_callback: CheckpointCallback instance
        model_dir: Directory to save the model
        train_id: Training ID string
        total_timesteps: Total number of timesteps to train
        monitor: Optional ModelEvaluationMonitor instance

    Returns:
        Path to the final model
    """
    exit_trainer = False

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=100
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        exit_trainer = True
    finally:
        # Save final model
        final_path = os.path.join(model_dir, f"{train_id}_final")
        model.save(final_path)
        print(f"Final model saved: {final_path}")

        # Wait for the evaluation monitor to evaluate the final model
        if monitor:
            print("Waiting for final evaluation to complete...")
            time.sleep(5)  # Give the monitor time to detect and evaluate the final model
            monitor.stop()

        env.close()

    print(f"Training completed with ID: {train_id}")
    print(f"Evaluation results saved to: {os.path.join(model_dir, f'{train_id}_eval_results.txt')}")
    print(f"Evaluation plot saved to: {os.path.join(model_dir, f'{train_id}_eval_plot.png')}")

    return final_path, exit_trainer
