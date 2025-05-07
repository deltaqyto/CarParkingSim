import os
import random

from AI.train_utils import find_td3_models, get_best_model, evaluate_model
from Simulation.simulation_wrapper import SimulationWrapper


def test_model(environment_factory):
    """User interface for testing a model."""
    print("\n======= TD3 Car Navigation Tester =======\n")
    print("Tip: Press 'r' to skip to the next episode\n")

    search_path = "models"

    # Get model ID
    while True:
        model_code = input("\nEnter model ID to test: ").strip().upper()
        existing_models = find_td3_models(search_path, model_code)
        if not existing_models:
            print(f"No models found with ID {model_code}. Please try again.")
            continue
        print(f"Found {len(existing_models)} checkpoints for model td3_{model_code}")
        break

    # Select iteration
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

    # Get test parameters
    while True:
        try:
            num_episodes = int(input("\nNumber of test episodes (default: 5): ") or "5")
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

    def create_wrapped_env():
        return SimulationWrapper(environment_factory, 0, seed)

    results = evaluate_model(
        environment=create_wrapped_env,
        model_path=model_path,
        num_episodes=num_episodes
    )

    if results:
        print("\n===== Test Results =====")
        print(f"Model: {model_path}")

        for reason, rate in results.get('stop_reasons', {}).items():
            if reason == 'Goal Hit':
                success_count = int(rate * num_episodes)
                print(f"Success Rate: {rate:.2f} ({success_count}/{num_episodes})")
            elif reason == 'Collision':
                collision_count = int(rate * num_episodes)
                print(f"Collision Rate: {rate:.2f} ({collision_count}/{num_episodes})")
            elif reason == 'Timeout':
                timeout_count = int(rate * num_episodes)
                print(f"Timeout Rate: {rate:.2f} ({timeout_count}/{num_episodes})")
            else:
                print(f"{reason} Rate: {rate:.2f}")

        print(f"Average Steps: {results.get('avg_steps', 0):.1f}")
        print(f"Average Reward: {results.get('avg_reward', 0):.2f}")
        print(f"Average Goal Distance: {results.get('average_goal_distance', 0):.2f}")
