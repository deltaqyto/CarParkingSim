from scripts.replot_training import test_model_interactive
from math import pi
from Simulation.environments import get_basic_env


def main():
    """Main entry point for the model tester."""
    # Dynamically import available environments
    available_envs = []

    try:

        available_envs.append(("Delta", get_basic_env, {"render": False, "goal_size": 2, "angle_tolerance": pi / 4}))
    except ImportError:
        pass

    # Add more environments as they become available

    if not available_envs:
        print("No environments found. Please check your installation.")
        return

    # Select environment
    print("\n======= TD3 Model Tester =======\n")
    print("Available environments:")
    for i, (name, _, _) in enumerate(available_envs):
        print(f"{i + 1}. {name}")

    while True:
        try:
            selection = int(input("\nSelect environment (number): ").strip())
            if 1 <= selection <= len(available_envs):
                env_name, env_factory, env_params = available_envs[selection - 1]
                print(f"Selected environment: {env_name}")
                break
            print(f"Please enter a number between 1 and {len(available_envs)}")
        except ValueError:
            print("Please enter a valid number.")

    # Start interactive testing
    test_model_interactive(env_factory, env_params)


if __name__ == "__main__":
    main()
