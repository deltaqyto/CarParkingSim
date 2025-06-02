from AI.test_model import test_model

if __name__ == "__main__":
    from modules.parking_modules import parking_env2

    base_env = parking_env2(render=False, goal_size=1, episodes=200, decimation=0.3, angle_tolerance=3.1415 / 6, generate_curb=True, spawn_probability=0, vision=False, vision_model='VIS_006b')

    # Run the test model function with the environment factory
    test_model(base_env)
