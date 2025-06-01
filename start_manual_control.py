import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pyinstrument import Profiler
from Simulation.environments import get_basic_env


def select_environment():
    """Let user choose which environment to run"""
    print("Select environment:")
    print("1. Basic Environment (default)")
    print("2. Parking Environment")

    out = None
    while out is None:
        choice = input("Enter your choice (number or press Enter for default): ").strip()

        if choice == "" or choice == "1":
            out = get_basic_env(render=True, goal_size=2, angle_tolerance=1.57)()
        else:
            print("Invalid choice. Please enter a number.")
    print("Loading Environment...")
    return out


if __name__ == "__main__":
    render = True
    instrument = False

    # Let user select environment
    sim_env = select_environment()

    print('=' * 20 + " Digest " + '=' * 20)
    print(sim_env.get_digest())
    print('=' * 20 + " End Digest " + '=' * 20)
    print("\nControls:")
    print("Arrow Keys: Drive (Up=Forward, Down=Reverse, Left/Right=Steer)")
    print("R: Reset environment")
    print("Q: Quit")
    print("-" * 50)

    rewards = 0

    # Create profiler
    if instrument:
        profiler = Profiler()
        profiler.start()

    # Number of steps to run
    num_steps = 10000
    step_count = 0

    while True:
        if instrument and step_count >= num_steps:
            break

        throttle = 0
        steer = 0
        if render and not instrument:
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_DOWN]:
                throttle = -1.0
            if keys[pygame.K_LEFT]:
                steer = -1.0
            if keys[pygame.K_RIGHT]:
                steer = 1.0
            if keys[pygame.K_r]:
                sim_env.reset_environment()
                throttle = 0
                steer = 0
                print("Environment reset!")
            if keys[pygame.K_q]:
                break

        done, observation, reward, state = sim_env.step([throttle, steer])
        rewards += reward
        step_count += 1

        car_position = state['car']['position']
        car_angle = state['car']['angle']
        print(f"\rCar Position: ({car_position[0]:.2f}, {car_position[1]:.2f}) Angle: {car_angle:.2f}°", end='')

        if done:
            print(f"\nEpisode reward: {rewards}")
            rewards = 0
            sim_env.reset_environment()
            print("Stop reasons:", state['stop_reasons'])
        if 'User Quit' in state['stop_reasons']:
            break

    # Stop profiling and print results
    if instrument:
        profiler.stop()

        # Print to console
        print(profiler.output_text(unicode=True, color=True))

        # Generate HTML report
        profiler.write_html("profile_report.html")
        print("Detailed HTML profile saved to 'profile_report.html'")
