import pygame
import numpy as np
from math import pi
from Objects.obstacles import RectObstacle

"""Collision isnt working"""
class StaticObstacle:
   
    def __init__(self, position, size=None, angle=None, direction=0, width=2, length=4.7, 
                 color=(0, 100, 200), console_logger=None):
        # Handle different constructor styles
        if size is not None:
            # If size is provided, assume it's (length, width)
            length, width = size
            
        # If angle is provided, use it instead of direction
        if angle is not None:
            direction = angle
            
        # Validate color is not red
        if color == (255, 0, 0) or (isinstance(color, tuple) and len(color) == 3 and 
                                   color[0] > 200 and color[1] < 50 and color[2] < 50):
            print("Warning: Red color not allowed for StaticObstacle. Using blue instead.")
            color = (0, 100, 200)  # Default to blue if red is attempted
            
        self.position = position if position is not None else [0, 0]
        self.direction = direction  # angle in degrees
        self.direction_vector = self._angular_direction_to_vector(direction)
        self.width = width
        self.length = length
        self.color = color
        
    def _angular_direction_to_vector(self, angle):
        """Convert angle in degrees to a normalized direction vector"""
        angle_rad = np.radians(angle)
        return [np.cos(angle_rad), np.sin(angle_rad)]
    
    def get_collision_rect(self):
        """Return collision rectangle data (center_point, size, angle)"""
        angle = -np.arctan2(self.direction_vector[1], self.direction_vector[0]) * 180 / pi
        return self.position, (self.length, self.width), -angle
    
    def get_aabb(self):
        """Get axis-aligned bounding box"""
        return RectObstacle(*self.get_collision_rect()).get_aabb()
    
    def get_corners(self):
        """Get the four corners of the obstacle"""
        return RectObstacle(*self.get_collision_rect()).get_corners()
    
    def draw(self, surface, transform_matrix):
        """Draw static obstacle on the surface"""
        wheel_width_offset = 1.4  # Same as Car class
        wheel_length_offset = 0.1
        
        # Calculate car dimensions and orientation
        car_angle = -np.arctan2(self.direction_vector[1], self.direction_vector[0]) * 180 / pi
        
        # Apply transform to position
        pos_vec = np.array([self.position[0], self.position[1], 1])
        screen_pos = transform_matrix @ pos_vec
        screen_x, screen_y = int(screen_pos[0]), int(screen_pos[1])
        
        # Determine dimensions after transform
        scale_x = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
        scale_y = np.sqrt(transform_matrix[1, 0] ** 2 + transform_matrix[1, 1] ** 2)
        car_width_px = int(self.width * scale_x)
        car_length_px = int(self.length * scale_y)
        
        # Calculate wheel dimensions
        wheel_width = max(3, int(car_width_px * 0.25))
        wheel_length = max(5, int(car_length_px * 0.2))
        
        # Calculate extended surface size for wheels
        extended_width = car_width_px + wheel_width * 2 * max(0.0, wheel_width_offset - 1.0)
        
        # Create car body rectangle
        car_rect = pygame.Surface((car_length_px, extended_width), pygame.SRCALPHA)
        
        # Calculate offset to center car
        width_offset = int(wheel_width * max(0.0, wheel_width_offset - 1.0))
        
        # Draw car body
        pygame.draw.rect(car_rect, self.color,
                         (0, width_offset, car_length_px, car_width_px), 0, 3)
        
        # Add direction triangle
        triangle_height = car_length_px // 4
        pygame.draw.polygon(car_rect, (50, 50, 50), [
            (car_length_px, width_offset + car_width_px // 2),
            (car_length_px - triangle_height, width_offset + car_width_px // 3),
            (car_length_px - triangle_height, width_offset + car_width_px * 2 // 3)
        ])
        
        # Wheel color
        wheel_color = (30, 30, 30)
        
        # For width positioning (account for the width_offset)
        left_wheel_y = width_offset - (wheel_width_offset - 1.0) * wheel_width
        right_wheel_y = width_offset + car_width_px - wheel_width + (wheel_width_offset - 1.0) * wheel_width
        
        # For length positioning
        rear_wheel_x = int(car_length_px * wheel_length_offset)
        front_wheel_x = int(car_length_px * (1.0 - wheel_length_offset)) - wheel_length
        
        # Draw all four wheels (static, no steering)
        pygame.draw.rect(car_rect, wheel_color,
                         (rear_wheel_x, left_wheel_y, wheel_length, wheel_width))
        pygame.draw.rect(car_rect, wheel_color,
                         (rear_wheel_x, right_wheel_y, wheel_length, wheel_width))
        pygame.draw.rect(car_rect, wheel_color,
                         (front_wheel_x, left_wheel_y, wheel_length, wheel_width))
        pygame.draw.rect(car_rect, wheel_color,
                         (front_wheel_x, right_wheel_y, wheel_length, wheel_width))
        
        # Rotate the entire car
        rotated_car = pygame.transform.rotate(car_rect, car_angle)
        
        # Get the new rectangle and position it
        car_pos_rect = rotated_car.get_rect(center=(screen_x, screen_y))
        
        # Draw to surface
        surface.blit(rotated_car, car_pos_rect)
    
    def calculate_aabb(self):
        """Calculate axis-aligned bounding box from corners"""
        corners = self.get_corners()
        
        min_x = min(x for x, y in corners)
        max_x = max(x for x, y in corners)
        min_y = min(y for x, y in corners)
        max_y = max(y for x, y in corners)
        return min_x, min_y, max_x, max_y
    
    def get_digest(self):
        """Return a string representation of the obstacle"""
        return (f"StaticObstacle(position={self.position}, direction={self.direction}, "
                f"color={self.color}, width={self.width}, length={self.length})")
                
    def reset(self, state=None):
        """Reset method for compatibility with environment modules"""
        # Nothing to reset for static obstacle
        pass
        
    def get_unified_state(self):
        """Return unified state for compatibility with environment modules"""
        return {
            'obstacles': [self]
        }
        
    def render(self, surface, transform_matrix):
        """Render method for compatibility with environment modules"""
        self.draw(surface, transform_matrix)


if __name__ == "__main__":
    # Test code
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Static Obstacle Test")
    clock = pygame.time.Clock()
    
    # Create static obstacles
    obstacles = [
        StaticObstacle(position=[0, 0], direction=0, color=(0, 100, 200)),
        StaticObstacle(position=[10, 5], direction=45, color=(0, 200, 0)),
        StaticObstacle(position=[-10, -5], direction=90, color=(200, 200, 0)),
        # Test with red color (should be replaced with blue)
        StaticObstacle(position=[5, -10], direction=135, color=(255, 0, 0))
    ]
    
    # Create a transform matrix (scale and translate)
    scale = 10  # pixels per meter
    transform = np.array([
        [scale, 0, 400],  # x scale, y shear, x translate
        [0, scale, 300],  # x shear, y scale, y translate
        [0, 0, 1]  # perspective
    ])
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Clear screen
        screen.fill((200, 200, 200))
        
        # Draw the obstacles
        for obstacle in obstacles:
            obstacle.draw(screen, transform)
        
        # Draw origin reference point
        origin = transform @ np.array([0, 0, 1])
        pygame.draw.circle(screen, (0, 0, 0), (int(origin[0]), int(origin[1])), 5)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()