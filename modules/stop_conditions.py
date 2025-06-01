import random
from math import pi, cos, sin, sqrt
import numpy as np
import pygame

from modules.generic_modules import GenericStop


class GenericGoal(GenericStop):
    def __init__(self, angle_tolerance, bidirectional=False, region=None, goal_size=2, angle_range=(-pi, pi),
                 goal_distance_range=(5, 100), name="GenericGoal"):
        self.goal_position = [0, 0]
        self.goal_angle = 0
        self.angle_tolerance = angle_tolerance
        self.bidirectional = bidirectional
        self.goal_size = goal_size
        self.goal_distance_range = goal_distance_range

        self.region = region if region is not None else [-30, 30, -20, 20]
        self.angle_range = angle_range
        self.name = name

    def reset(self, mode, state, region=None, angle_range=None):
        if region is not None:
            self.region = region
        if angle_range is not None:
            self.angle_range = angle_range

        self.pick_goal()
        dist = sqrt((state['car']['position'][0] - self.goal_position[0]) ** 2 +
                    (state['car']['position'][1] - self.goal_position[1]) ** 2)
        while dist < self.goal_distance_range[0] or dist > self.goal_distance_range[1]:
            self.pick_goal()
            dist = sqrt((state['car']['position'][0] - self.goal_position[0]) ** 2 +
                        (state['car']['position'][1] - self.goal_position[1]) ** 2)

        self.goal_angle = random.uniform(*self.angle_range)

    def pick_goal(self):
        self.goal_position = [random.uniform(self.region[0], self.region[1]),
                              random.uniform(self.region[2], self.region[3])]

    def get_digest(self):
        return (f"{self.name}(angle_tolerance={round(self.angle_tolerance, 2)}, bidirectional={self.bidirectional}, region={self.region}, "\
                f"angle_range={[round(x, 2) for x in self.angle_range]}, goal_size={self.goal_size}, "
                f"goal_distance_range={self.goal_distance_range})")

    def check_stop(self, state):
        position = state['car']['position']
        angle = state['car']['angle']

        if sqrt((position[0] - self.goal_position[0])**2 + (position[1] - self.goal_position[1])**2) > self.goal_size:
            return False, 'Nothing'

        angle_diff = min((angle - self.goal_angle) % (2 * pi), (self.goal_angle - angle) % (2 * pi))

        # If bidirectional, also check the opposite direction
        if self.bidirectional:
            opposite_goal = (self.goal_angle + pi) % (2 * pi)
            opposite_diff = min((angle - opposite_goal) % (2 * pi), (opposite_goal - angle) % (2 * pi))
            angle_diff = min(angle_diff, opposite_diff)

        if angle_diff < self.angle_tolerance:
            return True, 'Goal Hit'

        return False, 'Nothing'

    def render(self, screen, transform):
        goal_position_screen = transform @ np.append(self.goal_position, 1)

        origin_point = [0, 0]
        radius_point = [self.goal_size, 0]  # A point goal_size away from origin along x-axis

        # Transform points
        origin_screen = transform @ np.append(origin_point, 1)
        radius_screen = transform @ np.append(radius_point, 1)

        # Calculate scaled radius
        dx = radius_screen[0] - origin_screen[0]
        dy = radius_screen[1] - origin_screen[1]
        scaled_radius = int(np.sqrt(dx ** 2 + dy ** 2))

        # Draw goal circle
        pygame.draw.circle(
            screen,
            (0, 255, 0),  # Green color
            (int(goal_position_screen[0]), int(goal_position_screen[1])),
            scaled_radius,  # Use the scaled radius instead of fixed value
            0  # Filled circle
        )

        # Draw goal direction indicator
        if self.angle_tolerance < pi:
            # Calculate end point for direction line
            direction_length = 2  # Length of the direction indicator
            goal_direction = np.array([cos(self.goal_angle), sin(self.goal_angle)])
            direction_end = self.goal_position + goal_direction * direction_length
            direction_end_screen = transform @ np.append(direction_end, 1)

            # Draw line indicating orientation
            pygame.draw.line(
                screen,
                (255, 255, 0),  # Yellow color
                (int(goal_position_screen[0]), int(goal_position_screen[1])),
                (int(direction_end_screen[0]), int(direction_end_screen[1])),
                3  # Line width
            )

            # If double-sided, draw opposing direction line
            if self.bidirectional:
                goal_direction = np.array([cos(self.goal_angle), sin(self.goal_angle)])
                opposite_end = self.goal_position - goal_direction * direction_length
                opposite_end_screen = transform @ np.append(opposite_end, 1)

                pygame.draw.line(
                    screen,
                    (255, 255, 0),  # Yellow color
                    (int(goal_position_screen[0]), int(goal_position_screen[1])),
                    (int(opposite_end_screen[0]), int(opposite_end_screen[1])),
                    3  # Line width
                )

    def get_unified_state(self):
        return {'goals': [[*self.goal_position, self.goal_angle]],
                'bidirectional': self.bidirectional, 'goal_size': self.goal_size}


def omnidirectional_goal(**kwargs):
    return GenericGoal(angle_tolerance=2 * pi, name="OmniGoal", **kwargs)


def bidirectional_goal(angle_tolerance=0.1 * pi, **kwargs):
    return GenericGoal(angle_tolerance=angle_tolerance, bidirectional=True, name="BidirectionalGoal", **kwargs)


def directional_goal(angle_tolerance=0.25 * pi, **kwargs):
    return GenericGoal(angle_tolerance=angle_tolerance, name="DirectionalGoal", **kwargs)


class StepLimit(GenericStop):
    def __init__(self, step_limit=300):
        super().__init__()
        self.step_limit = step_limit

    def get_digest(self):
        return f"StepLimit(step_limit={self.step_limit})"

    def check_stop(self, state):
        if state['steps'] > self.step_limit:
            return True, 'Timeout'
        return False, 'Nothing'


class CollisionStop(GenericStop):
    def __init__(self):
        super().__init__()

    def get_digest(self):
        return f"CollisionStop()"

    def check_stop(self, state):
        if state['collisions']:
            return True, 'Collision'
        return False, 'Nothing'
