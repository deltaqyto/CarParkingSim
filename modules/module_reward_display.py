from modules.generic_modules import GenericReward
import pygame


class RewardDisplayModule(GenericReward):
    def __init__(self, max_history=100):
        super().__init__()
        self.max_history = max_history
        self.reward_history = []

        # Fonts will be initialized when pygame is available
        self.font_large = None
        self.font_small = None
        self.font_tiny = None

        # Colors
        self.bg_color = (0, 0, 0, 180)
        self.text_color = (255, 255, 255)
        self.positive_color = (0, 255, 0)
        self.negative_color = (255, 0, 0)

        # State data for rendering
        self.screen_width = 800
        self.screen_height = 600
        self.rendering_enabled = False
        self.current_total_reward = 0
        self.current_step_reward = 0
        self.current_reward_breakdown = {}
        self.current_car_position = [0, 0]
        self.current_step_count = 0

    def reset(self, mode, state):
        if mode == 'reward':
            print(state)
            self.reward_history = []
            self.current_total_reward = 0
            self.current_step_reward = 0
            self.current_reward_breakdown = {}
            self.current_step_count = 0

            # Check if rendering is enabled
            self.rendering_enabled = state['rendering']

            # Initialize fonts if rendering is enabled
            if self.rendering_enabled:
                self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
                self.font_small = pygame.font.SysFont('Arial', 18)
                self.font_tiny = pygame.font.SysFont('Arial', 14)

    def get_reward(self, state):
        # Check if rendering is enabled
        self.rendering_enabled = state['rendering']

        # Update screen dimensions
        self.screen_width, self.screen_height = state['screen_size']

        # Get reward breakdown from state
        reward_types = state.get('reward_types', {})
        self.current_reward_breakdown = reward_types

        # Calculate current step reward
        if reward_types:
            self.current_step_reward = sum(reward_types.values())
            self.add_reward(self.current_step_reward)
            self.current_total_reward += self.current_step_reward

        # Get car position and step count
        self.current_car_position = state['car']['position']

        # Step count would need to be tracked by the simulation environment
        # For now, we'll increment it ourselves
        self.current_step_count += 1

        return None, 0

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

    def render(self, screen, transform_matrix):
        """Render reward display on screen"""
        if not self.rendering_enabled:
            return

        # Update screen dimensions from actual screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        # Initialize fonts if not done yet
        if self.font_large is None:
            self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
            self.font_small = pygame.font.SysFont('Arial', 18)
            self.font_tiny = pygame.font.SysFont('Arial', 14)

        self._draw_reward_panel(screen)

    def _draw_reward_panel(self, surface):
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
        total_color = self.positive_color if self.current_total_reward > 0 else self.negative_color if self.current_total_reward < 0 else self.text_color
        total_text = self.font_large.render(f"Total: {self.current_total_reward:.2f}", True, total_color)
        surface.blit(total_text, (panel_x + 10, y_offset))
        y_offset += 30

        # Current step reward
        current_color = self.positive_color if self.current_step_reward > 0 else self.negative_color if self.current_step_reward < 0 else self.text_color
        current_text = self.font_small.render(f"Step: {self.current_step_reward:.3f}", True, current_color)
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

        for reward_name, reward_value in self.current_reward_breakdown.items():
            if reward_value != 0:  # Only show non-zero rewards
                reward_color = self.positive_color if reward_value > 0 else self.negative_color
                reward_text = self.font_tiny.render(f"{reward_name}: {reward_value:.3f}", True, reward_color)
                surface.blit(reward_text, (panel_x + 15, y_offset))
                y_offset += 16

        # Car info
        y_offset += 10
        car_info = self.font_tiny.render(f"Pos: ({self.current_car_position[0]:.1f}, {self.current_car_position[1]:.1f})", True, self.text_color)
        surface.blit(car_info, (panel_x + 10, y_offset))
        y_offset += 16

        step_info = self.font_tiny.render(f"Step: {self.current_step_count}", True, self.text_color)
        surface.blit(step_info, (panel_x + 10, y_offset))

    def get_digest(self):
        return f"RewardDisplayModule(max_history={self.max_history})"
