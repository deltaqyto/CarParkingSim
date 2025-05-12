import os
import random
import glob
import time
import multiprocessing
import numpy as np

from AI.train_utils import find_td3_models, get_best_model, evaluate_model, get_timestep_from_path
from Simulation.simulation_wrapper import SimulationWrapper
from Utility.plotting import plot_evaluation_results


def test_single_model(model_path, environment_factory, env_params, seed=None, num_episodes=20):
    """
    Test a single model checkpoint.

    Args:
        model_path: Path to the model checkpoint
        environment_factory: Function to create environment
        env_params: Params for it
        seed: Random seed
        num_episodes: Number of episodes to test

    Returns:
        Evaluation results dictionary
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(0, 10000)

    def create_wrapped_env():
        return SimulationWrapper(environment_factory(**env_params), 0, seed)

    results = evaluate_model(
        environment=create_wrapped_env,
        model_path=model_path,
        num_episodes=num_episodes
    )

    return results


def _test_model_process(model_path, environment_factory, env_params, seed, num_episodes, result_queue):
    try:
        result = test_single_model(model_path, environment_factory, env_params, seed, num_episodes)
        if result:
            timestep = get_timestep_from_path(model_path)
            print(f"Tested {os.path.basename(model_path)} - "
                  f"Reward: {result['avg_reward']:.2f}, Steps: {result['avg_steps']:.1f}")
            result_queue.put(result)
    except Exception as e:
        print(f"Error testing {os.path.basename(model_path)}: {e}")
        result_queue.put(None)


def test_all_checkpoints(model_code, environment_factory, env_params, search_path="models", num_episodes=20, seed=None):
    """
    Test all checkpoints of a training run in parallel.

    Args:
        model_code: Model ID code
        environment_factory: Function that creates the environment
        env_params: Parameters for the env
        search_path: Base directory for models
        num_episodes: Number of episodes to test for each checkpoint
        seed: Random seed (if None, a random seed will be used)

    Returns:
        List of evaluation results for all checkpoints
    """
    # Find all checkpoint models
    model_dir = os.path.join(search_path, f"td3_{model_code}")
    checkpoint_pattern = os.path.join(model_dir, f"*.zip")
    model_files = glob.glob(checkpoint_pattern)

    if not model_files:
        print(f"No models found for {model_code}")
        return []

    print(f"Found {len(model_files)} checkpoints for model td3_{model_code}")

    # Sort models by timestep
    model_files.sort(key=get_timestep_from_path)

    # Set random seed if not provided
    if seed is None:
        seed = random.randint(0, 10000)
    print(f"Using random seed: {seed}")

    start_time = time.time()

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    max_processes = min(14, multiprocessing.cpu_count())
    print(f"Using {max_processes} processes for parallel evaluation")

    for i in range(0, len(model_files), max_processes):
        batch = model_files[i:i + max_processes]
        processes = []

        for model_path in batch:
            p = multiprocessing.Process(
                target=_test_model_process,
                args=(model_path, environment_factory, env_params, seed, num_episodes, result_queue)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    all_results = []
    while not result_queue.empty():
        result = result_queue.get()
        if result:
            all_results.append(result)

    # Sort results by timestep
    all_results.sort(key=lambda x: x['timestep'])

    elapsed_time = time.time() - start_time
    print(f"Completed testing {len(all_results)} models in {elapsed_time:.1f} seconds")

    return all_results


def test_model_interactive(environment_factory, env_params):
    print("\n======= TD3 Model Tester =======\n")
    print("Tip: Press 'r' to skip to the next episode\n")

    search_path = "models"

    # Get model ID
    while True:
        model_code = input("\nEnter model ID to test: ").strip().upper()
        if not model_code:
            print("Model ID cannot be empty. Please try again.")
            continue

        existing_models = find_td3_models(search_path, model_code)
        if not existing_models:
            print(f"No models found with ID {model_code}. Please try again.")
            continue

        print(f"Found {len(existing_models)} checkpoints for model td3_{model_code}")
        break

    # Ask if user wants to test all checkpoints or a specific one
    # Default is now to test all checkpoints
    while True:
        test_all = input("\nDo you want to test all checkpoints? (y/n, default: y): ").strip().lower()

        if not test_all:
            test_all = 'y'

        if test_all in ['y', 'n']:
            break

        print("Please enter 'y' for yes or 'n' for no.")

    if test_all == 'y':
        # Get number of episodes (default is now 20)
        while True:
            try:
                episodes_input = input("\nNumber of test episodes per checkpoint (default: 20): ").strip()

                if not episodes_input:
                    num_episodes = 20
                    break

                num_episodes = int(episodes_input)
                if num_episodes > 0:
                    break

                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        # Generate a random seed
        seed = random.randint(0, 10000)
        print(f"Using random seed: {seed}")

        # Test all checkpoints in parallel
        print(f"\nTesting all checkpoints for model td3_{model_code}...")
        results = test_all_checkpoints(
            model_code=model_code,
            environment_factory=environment_factory,
            env_params=env_params,
            search_path=search_path,
            num_episodes=num_episodes,
            seed=seed
        )

        if results:
            # Generate plot with _retest suffix to not overwrite original
            model_dir = os.path.join(search_path, f"td3_{model_code}")
            plot_path = os.path.join(model_dir, f"{model_code}_eval_plot_retest.png")
            plot_evaluation_results(
                results=results,
                save_path=plot_path,
                title=f'Model Evaluation Results - {model_code} (Retest)'
            )
            print(f"\nEvaluation plot saved to: {plot_path}")

            # Save results to file
            results_path = os.path.join(model_dir, f"{model_code}_eval_results_retest.txt")
            with open(results_path, 'w') as f:
                f.write(f"Retest Evaluation Results for {model_code}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestep, Average reward, Avg Steps, Stop reasons, Rewards, Average Goal distances, Nearest Goal Distances\n")
                f.write("-" * 80 + "\n")
                for result in results:
                    f.write(f"{result}\n")
            print(f"Detailed results saved to: {results_path}")

    else:
        # Select specific iteration for testing
        while True:
            selected_iteration = input("\nEnter specific iteration to load (or leave blank for best available): ").strip()

            if not selected_iteration:
                model_path = get_best_model(existing_models)
                print(f"Using best available model: {os.path.basename(model_path)}")
                break

            # Handle final model
            if selected_iteration.lower() == "final":
                model_path = os.path.join(search_path, f"td3_{model_code}", f"{model_code}_final.zip")
                if os.path.exists(model_path):
                    print(f"Selected model: {model_path}")
                    break
                else:
                    print(f"Final model not found: {model_path}. Try again.")
                    continue

            # Handle steps format
            if selected_iteration.isdigit():
                selected_iteration = f"{model_code}_{selected_iteration}_steps.zip"
            elif not selected_iteration.endswith('.zip'):
                selected_iteration = f"{selected_iteration}.zip"

            model_path = os.path.join(search_path, f"td3_{model_code}", selected_iteration)
            if os.path.exists(model_path):
                print(f"Selected model: {model_path}")
                break
            else:
                print(f"Model not found: {model_path}. Try again.")

        # Get number of episodes (default is now 20)
        while True:
            try:
                episodes_input = input("\nNumber of test episodes (default: 20): ").strip()

                if not episodes_input:
                    num_episodes = 20
                    break

                num_episodes = int(episodes_input)
                if num_episodes > 0:
                    break

                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        # Run the test
        print(f"\nTesting model: {os.path.basename(model_path)}")
        print(f"Running {num_episodes} episodes...\n")

        # Generate a random seed
        seed = random.randint(0, 10000)
        print(f"Using random seed: {seed}")

        results = test_single_model(
            model_path=model_path,
            environment_factory=environment_factory,
            env_params=env_params,
            seed=seed,
            num_episodes=num_episodes
        )

        if results:
            print("\n===== Test Results =====")
            print(f"Model: {model_path}")

            # Print all stop reasons (generalized)
            print("\nStop Reasons:")
            for reason, rate in results.get('stop_reasons', {}).items():
                count = int(rate * num_episodes)
                print(f"  {reason}: {rate:.2f} ({count}/{num_episodes})")

            print(f"\nAverage Steps: {results.get('avg_steps', 0):.1f}")
            print(f"Average Reward: {results.get('avg_reward', 0):.2f}")
            print(f"Average Goal Distance: {results.get('average_goal_distance', 0):.2f}")
            print(f"Nearest Goal Distance: {results.get('nearest_goal_distance', 0):.2f}")

            # Print reward breakdown
            print("\nReward Breakdown:")
            for component, value in results.get('reward_breakdown', {}).items():
                print(f"  {component}: {value:.2f}")


def test_multiple_checkpoints(settings):
    search_path = 'models'
    seed = random.randint(0, 1000000)
    for setting in settings:
        env, param, model, iterations, test_all = setting
        model = model.strip().upper()
        results = test_all_checkpoints(
            model_code=model,
            environment_factory=env,
            env_params=param,
            search_path=search_path,
            num_episodes=iterations,
            seed=seed
        )

        if results:
            # Generate plot with _retest suffix to not overwrite original
            model_dir = os.path.join(search_path, f"td3_{model}")
            plot_path = os.path.join(model_dir, f"{model}_eval_plot_retest.png")
            plot_evaluation_results(
                results=results,
                save_path=plot_path,
                title=f'Model Evaluation Results - {model} (Retest)'
            )
            print(f"\nEvaluation plot saved to: {plot_path}")

            # Save results to file
            results_path = os.path.join(model_dir, f"{model}_eval_results_retest.txt")
            with open(results_path, 'w') as f:
                f.write(f"Retest Evaluation Results for {model}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestep, Average reward, Avg Steps, Stop reasons, Rewards, Average Goal distances, Nearest Goal Distances\n")
                f.write("-" * 80 + "\n")
                for result in results:
                    f.write(f"{result}\n")
            print(f"Detailed results saved to: {results_path}")
