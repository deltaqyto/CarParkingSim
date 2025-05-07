import pygame
import numpy as np


class CollisionSystem:
    def __init__(self):
        self.collidable_objects = []

    def add_object(self, obj):
        """Add an object to collision detection system"""
        self.collidable_objects.append(obj)

    def remove_object(self, obj):
        """Remove an object from collision detection system"""
        if obj in self.collidable_objects:
            self.collidable_objects.remove(obj)

    def collide_against(self, obj):
        """Check collisions between specified object and all other objects"""
        collisions = []

        target_aabb = obj.get_aabb()

        for other in self.collidable_objects:
            if other is obj:
                continue  # Skip comparing with itself

            # Calculate AABB for the other object
            other_aabb = other.get_aabb()

            # Quick AABB overlap test
            if not self._aabb_overlap(target_aabb, other_aabb):
                continue  # No collision possible, skip detailed check

            # Run expensive SAT test
            if self._check_collision_sat(obj, other):
                collisions.append(other)

        return collisions

    @staticmethod
    def _aabb_overlap(aabb1, aabb2):
        """Check if two AABBs overlap"""
        min_x1, min_y1, max_x1, max_y1 = aabb1
        min_x2, min_y2, max_x2, max_y2 = aabb2

        return not (max_x1 < min_x2 or
                    min_x1 > max_x2 or
                    max_y1 < min_y2 or
                    min_y1 > max_y2)

    def _check_collision_sat(self, obj1, obj2):
        """
        Check collision using Separating Axis Theorem (SAT)
        This handles rotated rectangles accurately
        """
        # Get corner points of first rectangle
        rect1_points = obj1.get_corners()

        # Get corner points of second rectangle
        rect2_points = obj2.get_corners()

        # Get the axes to check (normals to each rectangle's sides)
        axes = self._get_axes(rect1_points, rect2_points)

        # Check for separation along each axis
        for axis in axes:
            # Project rectangles onto the axis
            proj1 = self._project_onto_axis(rect1_points, axis)
            proj2 = self._project_onto_axis(rect2_points, axis)

            # If projections don't overlap, rectangles don't collide
            if not self._overlap(proj1, proj2):
                return False

        # No separating axis found, rectangles collide
        return True


    @staticmethod
    def _get_axes(rect1_points, rect2_points):
        """Get the axes to test for the SAT algorithm"""
        axes = []

        # Add axes from rect1 (normals to each edge)
        for i in range(4):
            p1 = rect1_points[i]
            p2 = rect1_points[(i + 1) % 4]

            # Edge vector
            edge = (p2[0] - p1[0], p2[1] - p1[1])

            # Normal to edge (perpendicular)
            normal = (-edge[1], edge[0])

            # Normalize
            length = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if length > 0:
                normal = (normal[0] / length, normal[1] / length)
                axes.append(normal)

        # Add axes from rect2
        for i in range(4):
            p1 = rect2_points[i]
            p2 = rect2_points[(i + 1) % 4]

            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])

            length = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if length > 0:
                normal = (normal[0] / length, normal[1] / length)
                axes.append(normal)

        return axes

    @staticmethod
    def _project_onto_axis(points, axis):
        """Project all points of a shape onto an axis"""
        min_proj = float('inf')
        max_proj = float('-inf')

        for point in points:
            # Dot product gives the projection
            projection = point[0] * axis[0] + point[1] * axis[1]

            min_proj = min(min_proj, projection)
            max_proj = max(max_proj, projection)

        return min_proj, max_proj

    @staticmethod
    def _overlap(proj1, proj2):
        """Check if two projections overlap"""
        return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

    def draw_debug(self, surface, transform_matrix):
        """Draw collision rectangles for debugging"""
        for obj in self.collidable_objects:
            corners = obj.get_corners()

            # Transform corners to screen coordinates
            screen_corners = []
            for x, y in corners:
                pos_vec = np.array([x, y, 1])
                screen_pos = transform_matrix @ pos_vec
                screen_corners.append((int(screen_pos[0]), int(screen_pos[1])))

            # Draw outline
            pygame.draw.lines(surface, (0, 255, 0), True, screen_corners, 2)
