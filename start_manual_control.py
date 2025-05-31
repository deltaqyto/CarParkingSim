import os

from Simulation.environments import get_yolo_env

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from os import path
import sys

from Objects.car import Car
from Utility.console_logger import ConsoleLogger
from Simulation.simulation_environment import SimulationEnvironment

from pyinstrument import Profiler # pip install pygame numpy pyinstrument


if __name__ == "__main__":
    render = True
    instrument = False

    from Simulation.environments import get_yolo_env
    sim_env = get_yolo_env(render, 2, 1.57)()
    print('=' * 20 + " Digest " + '=' * 20)
    print(sim_env.get_digest())
    print('=' * 20 + " End Digest " + '=' * 20)
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
            if keys[pygame.K_q]:
                break

        done, observation, reward, state = sim_env.step([throttle, steer])

        # Debug: Check if obstacles are in the final state
        # if step_count % 60 == 0:  # Every 60 frames
        #     total_obstacles = len(state.get('obstacles', []))
        #     print(f"DEBUG: Final state has {total_obstacles} obstacles")
            
        #     # Check collision system
        #     if hasattr(sim_env, 'collision_system'):
        #         print(f"DEBUG: Collision system exists")
        #         # Try to access collision objects if possible
        #     else:
        #         print("DEBUG: No collision system found!")
        
        #print(state)
        rewards += reward
        step_count += 1
        
        car_position = state['car']['position']
        car_angle = state['car']['angle']
        print(f"\rCar Position: ({car_position[0]:.2f}, {car_position[1]:.2f}) Angle: {car_angle:.2f}°", end='')


        if done:
            print(f"Episode reward: {rewards}")
            rewards = 0
            sim_env.reset_environment()
            print(state['stop_reasons'])
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