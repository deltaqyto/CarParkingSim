import os
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def plot_stop_reasons(ax, timesteps, stop_reasons):
    """
    Plot stop reasons as separate lines.

    Args:
        ax: Matplotlib axes
        timesteps: List of training timesteps
        stop_reasons: List of dictionaries mapping stop reason to rate
    """
    # Get all unique stop reason keys across all dictionaries
    all_stop_reasons = set()
    for sr in stop_reasons:
        all_stop_reasons.update(sr.keys())

    # Plot each stop reason as its own line
    for reason in all_stop_reasons:
        reason_values = []
        for i, sr in enumerate(stop_reasons):
            value = sr.get(reason, 0)  # Default to 0 if key is missing
            reason_values.append((timesteps[i], value))
        if reason_values:
            x_vals, y_vals = zip(*reason_values)
            ax.plot(x_vals, y_vals, 'o-', label=reason)

    ax.set_xlabel('Training Timestep')
    ax.set_ylabel('Rate')
    ax.set_title('Stop Reasons per Episode')
    ax.legend()
    ax.grid(True)


def plot_steps_and_rewards(ax, timesteps, avg_steps, avg_rewards):
    """
    Plot average steps and rewards with dual y-axes.

    Args:
        ax: Matplotlib axes
        timesteps: List of training timesteps
        avg_steps: List of average steps per episode
        avg_rewards: List of average rewards per episode
    """
    ax2 = ax.twinx()  # Create a second y-axis

    # Plot average steps on left y-axis
    line1, = ax.plot(timesteps, avg_steps, 'bo-', label='Avg Steps')
    ax.set_xlabel('Training Timestep')
    ax.set_ylabel('Steps', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    # Plot average rewards on right y-axis
    line2, = ax2.plot(timesteps, avg_rewards, 'ro-', label='Avg Rewards')
    ax2.set_ylabel('Reward', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add combined legend
    ax.legend([line1, line2], ['Avg Steps', 'Avg Rewards'], loc='upper left')
    ax.set_title('Average Steps and Rewards per Episode')
    ax.grid(True)


def plot_goal_distances(ax, timesteps, goal_distances, final_goal_distances):
    """
    Plot average and final goal distances.

    Args:
        ax: Matplotlib axes
        timesteps: List of training timesteps
        goal_distances: List of average goal distances per episode
        final_goal_distances: List of final goal distances per episode
    """
    ax.plot(timesteps, goal_distances, 'mo-', label='Average Distance')
    ax.plot(timesteps, final_goal_distances, 'go-', label='Final Distance')
    ax.set_xlabel('Training Timestep')
    ax.set_ylabel('Goal Distance')
    ax.set_title('Distances to Goal per Episode')
    ax.legend(loc='upper left')
    ax.grid(True)


def plot_reward_breakdown(ax, timesteps, reward_breakdowns):
    """
    Plot reward breakdown components.

    Args:
        ax: Matplotlib axes
        timesteps: List of training timesteps
        reward_breakdowns: List of dictionaries mapping reward component to value
    """
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
                component_values.append((timesteps[i], rb[key]))
        if component_values:
            x_vals, y_vals = zip(*component_values)
            ax.plot(x_vals, y_vals, '-', label=key)

    ax.set_xlabel('Training Timestep')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Breakdown per Episode')
    ax.legend(loc='upper left')
    ax.grid(True)


def plot_evaluation_results(results, save_path, title='Model Evaluation Results'):
    # Filter out 'final' model for plotting by timestep
    plot_results = [r for r in results if r['timestep'] != float('inf')]
    if len(plot_results) < 1:
        # Create empty placeholder plot if no data
        plt.figure(figsize=(1, 1))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return  # Need at least 1 point to make a plot
    # Sort by timestep
    plot_results.sort(key=lambda x: x['timestep'])

    timesteps = [r['timestep'] for r in plot_results]
    avg_rewards = [r['avg_reward'] for r in plot_results]
    avg_steps = [r['avg_steps'] for r in plot_results]
    stop_reasons = [r['stop_reasons'] for r in plot_results]
    reward_breakdowns = [r['reward_breakdown'] for r in plot_results]
    goal_distances = [r['average_goal_distance'] for r in plot_results]
    final_goal_distances = [r['final_goal_distance'] for r in plot_results]

    # 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # Plot 1: Stop Reasons (top-left)
    plot_stop_reasons(axes[0, 0], timesteps, stop_reasons)

    # Plot 2: Average Steps & Rewards with dual y-axes (top-right)
    plot_steps_and_rewards(axes[0, 1], timesteps, avg_steps, avg_rewards)

    # Plot 3: Goal distances (bottom-left)
    plot_goal_distances(axes[1, 0], timesteps, goal_distances, final_goal_distances)

    # Plot 4: Reward breakdowns (bottom-right)
    plot_reward_breakdown(axes[1, 1], timesteps, reward_breakdowns)

    plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout()

    # Save figure without displaying
    plt.savefig(save_path)
    plt.close()  # Important! Close the figure to avoid memory leaks


def plot_custom_evaluation(results, save_path, subplot_config, figsize=(20, 10), title='Custom Evaluation Results'):
    """
    Create a custom evaluation plot with configurable subplots.

    Args:
        results: List of evaluation result dictionaries
        save_path: Path to save the plot
        subplot_config: List of tuples (plotting_function, subplot_position_dict)
            where plotting_function is a callable that takes (ax, results) and
            subplot_position_dict specifies the grid position
        figsize: Figure size (width, height) in inches
        title: Plot title
    """
    fig = plt.figure(figsize=figsize)

    for plot_func, subplot_pos in subplot_config:
        ax = fig.add_subplot(**subplot_pos)
        plot_func(ax, results)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid memory leaks
