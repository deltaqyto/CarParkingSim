import pygame
import numpy as np


class RectObstacle:
    """Optimized rectangular obstacle class with cached calculations"""

    def __init__(self, position, size, angle=0, color=(100, 100, 100)):
        self.position = position
        self.size = size
        self.angle = angle
        self.color = color

        # Cached values
        self._corners = None
        self._aabb = None
        self._need_update = True

    def update_position(self, position):
        """Update position and invalidate cache"""
        self.position = position
        self._need_update = True

    def update_angle(self, angle):
        """Update angle and invalidate cache"""
        self.angle = angle
        self._need_update = True

    def update_size(self, size):
        """Update size and invalidate cache"""
        self.size = size
        self._need_update = True

    def get_collision_rect(self):
        """Return collision rectangle data"""
        return self.position, self.size, self.angle

    def get_aabb(self):
        """Get cached AABB or calculate if needed"""
        if self._need_update or self._aabb is None:
            self._update_cache()
        return self._aabb

    def get_corners(self):
        """Get cached corners or calculate if needed"""
        if self._need_update or self._corners is None:
            self._update_cache()
        return self._corners

    def _update_cache(self):
        """Update cached values (corners and AABB)"""
        self._calculate_corners()
        self._calculate_aabb()
        self._need_update = False

    def _calculate_corners(self):
        """Calculate the four corners of the rectangle in world space"""
        # Convert angle to radians
        angle_rad = np.radians(self.angle)

        # Calculate half-dimensions
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2

        # Calculate corners in local space (before rotation)
        corners_local = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])

        # Rotation matrix (pre-calculate sin/cos)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        # Apply rotation and translation to all corners at once
        rotated_corners = corners_local @ rotation.T

        # Add position to all corners (vectorized)
        pos_array = np.array(self.position)
        self._corners = rotated_corners + pos_array

    def _calculate_aabb(self):
        """Calculate AABB directly from corners"""
        # If corners aren't calculated yet, compute them
        if self._corners is None:
            self._calculate_corners()

        # Get min/max x and y
        min_x = np.min(self._corners[:, 0])
        max_x = np.max(self._corners[:, 0])
        min_y = np.min(self._corners[:, 1])
        max_y = np.max(self._corners[:, 1])

        self._aabb = (min_x, min_y, max_x, max_y)

    def draw(self, surface, transform_matrix):
        """Draw obstacle on the surface"""
        # Calculate screen position
        pos_vec = np.array([self.position[0], self.position[1], 1])
        screen_pos = transform_matrix @ pos_vec

        # Determine obstacle dimensions after transform
        scale_x = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
        scale_y = np.sqrt(transform_matrix[1, 0] ** 2 + transform_matrix[1, 1] ** 2)

        # Scale the obstacle size
        scaled_width = int(self.size[0] * scale_x)
        scaled_height = int(self.size[1] * scale_y)

        # Create a surface for the obstacle
        max_dim = max(scaled_width, scaled_height) * 2
        obstacle_surf = pygame.Surface((max_dim, max_dim), pygame.SRCALPHA)

        # Create rectangle
        rect = pygame.Rect(0, 0, scaled_width, scaled_height)
        rect.center = (max_dim // 2, max_dim // 2)

        # Draw rectangle
        pygame.draw.rect(obstacle_surf, self.color, rect)

        # Rotate surface
        rotated = pygame.transform.rotate(obstacle_surf, -self.angle)

        # Position rotated surface
        rot_rect = rotated.get_rect(center=(screen_pos[0], screen_pos[1]))

        # Draw to surface
        surface.blit(rotated, rot_rect)
