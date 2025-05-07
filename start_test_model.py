from AI.test_model import test_model

if __name__ == "__main__":
    from Simulation.environments import get_basic_env

    base_env = get_basic_env(render=True, goal_size=1)

    # Run the test model function with the environment factory
    test_model(base_env)
