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


class RewardDisplay:
    """Helper class to display rewards on screen"""
    def __init__(self, screen_width, screen_height):
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.font_tiny = pygame.font.SysFont('Arial', 14)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Colors
        self.bg_color = (0, 0, 0, 180)  # Semi-transparent black
        self.text_color = (255, 255, 255)
        self.positive_color = (0, 255, 0)
        self.negative_color = (255, 0, 0)
        self.neutral_color = (255, 255, 0)
        
        # Track reward history for running average
        self.reward_history = []
        self.max_history = 100
        
    def add_reward(self, reward):
        """Add reward to history for running average"""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
    
    def get_running_average(self):
        """Get running average of recent rewards"""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)
    
    def draw_reward_panel(self, surface, total_reward, current_reward, reward_breakdown, car_position, step_count):
        """Draw the reward information panel"""
        panel_width = 300
        panel_height = 220
        panel_x = self.screen_width - panel_width - 10
        panel_y = 10
        
        # Create semi-transparent background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.bg_color)
        surface.blit(panel_surface, (panel_x, panel_y))
        
        y_offset = panel_y + 10
        
        # Total episode reward
        total_color = self.positive_color if total_reward > 0 else self.negative_color if total_reward < 0 else self.text_color
        total_text = self.font_large.render(f"Total: {total_reward:.2f}", True, total_color)
        surface.blit(total_text, (panel_x + 10, y_offset))
        y_offset += 30
        
        # Current step reward
        current_color = self.positive_color if current_reward > 0 else self.negative_color if current_reward < 0 else self.text_color
        current_text = self.font_small.render(f"Step: {current_reward:.3f}", True, current_color)
        surface.blit(current_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Running average
        avg_reward = self.get_running_average()
        avg_color = self.positive_color if avg_reward > 0 else self.negative_color if avg_reward < 0 else self.text_color
        avg_text = self.font_small.render(f"Avg: {avg_reward:.3f}", True, avg_color)
        surface.blit(avg_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Reward breakdown
        breakdown_title = self.font_small.render("Breakdown:", True, self.text_color)
        surface.blit(breakdown_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        for reward_name, reward_value in reward_breakdown.items():
            if reward_value != 0:  # Only show non-zero rewards
                reward_color = self.positive_color if reward_value > 0 else self.negative_color
                reward_text = self.font_tiny.render(f"{reward_name}: {reward_value:.3f}", True, reward_color)
                surface.blit(reward_text, (panel_x + 15, y_offset))
                y_offset += 16
        
        # Car info
        y_offset += 10
        car_info = self.font_tiny.render(f"Pos: ({car_position[0]:.1f}, {car_position[1]:.1f})", True, self.text_color)
        surface.blit(car_info, (panel_x + 10, y_offset))
        y_offset += 16
        
        step_info = self.font_tiny.render(f"Step: {step_count}", True, self.text_color)
        surface.blit(step_info, (panel_x + 10, y_offset))
    
    def draw_controls_help(self, surface):
        """Draw control instructions"""
        help_text = [
            "Controls:",
            "↑↓ - Throttle",
            "←→ - Steering", 
            "R - Reset",
            "Q - Quit"
        ]
        
        y_start = self.screen_height - (len(help_text) * 18) - 10
        
        for i, text in enumerate(help_text):
            color = self.neutral_color if i == 0 else self.text_color
            text_surface = self.font_tiny.render(text, True, color)
            surface.blit(text_surface, (10, y_start + i * 18))


if __name__ == "__main__":
    render = True
    instrument = False

    from Simulation.environments import get_yolo_env
    sim_env = get_yolo_env(render, 2, 1.57)()
    print('=' * 20 + " Digest " + '=' * 20)
    print(sim_env.get_digest())
    print('=' * 20 + " End Digest " + '=' * 20)
    
    # Initialize reward tracking
    total_rewards = 0
    current_reward = 0
    reward_breakdown = {}
    
    # Create reward display
    reward_display = RewardDisplay(sim_env.screen_width, sim_env.screen_height)

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
                total_rewards = 0
                reward_display.reward_history.clear()
                print("Environment reset!")
            if keys[pygame.K_q]:
                break

        done, observation, reward, state = sim_env.step([throttle, steer])
        
        # Update reward tracking
        current_reward = reward
        total_rewards += reward
        reward_breakdown = state.get('reward_types', {})
        reward_display.add_reward(reward)

        step_count += 1
        
        # Draw reward information on screen (after the normal rendering)
        if render and hasattr(sim_env, 'screen'):
            car_position = state['car']['position']
            reward_display.draw_reward_panel(
                sim_env.screen, 
                total_rewards, 
                current_reward, 
                reward_breakdown, 
                car_position, 
                step_count
            )
            reward_display.draw_controls_help(sim_env.screen)
            
            # Update the display after drawing our additions
            pygame.display.flip()
        
        car_position = state['car']['position']
        car_angle = state['car']['angle']
        print(f"\rCar Position: ({car_position[0]:.2f}, {car_position[1]:.2f}) Angle: {car_angle:.2f}°", end='')

        if done:
            print(f"\nEpisode reward: {total_rewards:.3f}")
            print(f"Final breakdown: {reward_breakdown}")
            total_rewards = 0
            reward_display.reward_history.clear()
            sim_env.reset_environment()
            print(f"Stop reasons: {state['stop_reasons']}")
            
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