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

    def render(self, screen, transform_matrix):
        # Render the YOLO goals as circles
        for goal_x, goal_y, goal_angle in self.goals_from_yolo:
            # Transform goal position to screen coordinates
            goal_screen = transform_matrix @ np.array([goal_x, goal_y, 1])
            goal_screen_pos = (int(goal_screen[0]), int(goal_screen[1]))
            
            # Calculate radius in screen coordinates - make even smaller
            radius_world = self.goal_radius * 0.5  # Make visual radius even smaller
            radius_screen = max(3, int(radius_world * transform_matrix[0, 0]))  # Minimum 3 pixels
            
            # Draw goal circle (green with red border) - smaller
            pygame.draw.circle(screen, (0, 255, 0), goal_screen_pos, radius_screen, 2)
            pygame.draw.circle(screen, (255, 0, 0), goal_screen_pos, radius_screen, 1)

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


from modules.generic_modules import GenericStop
import pygame
import numpy as np


class YOLOGoalStop(GenericStop):
    def __init__(self, goal_radius=0.4):
        super().__init__()
        self.goal_radius = goal_radius
        self.goals_from_yolo = []

    def reset(self, mode, state=None):
        # Extract goals from YOLO detector in environment modules
        self.goals_from_yolo = []
        
        if state and 'environment' in state:
            for module_state in state['environment']:
                if module_state.get('name') == 'YOLOGoals':
                    # Get the formatted goals (x, y, angle) from YOLO detector
                    yolo_goals = module_state.get('goals', [])
                    self.goals_from_yolo = yolo_goals
                    print(f"YOLOGoalStop: Found {len(yolo_goals)} YOLO goals")
                    break
        
        # Don't raise error if no goals found initially - YOLO needs time to detect
        if not self.goals_from_yolo:
            print("YOLOGoalStop: No YOLO goals found initially, will wait for detection")

    def check_stop(self, state):
        # Update goals from current state (in case YOLO detected new ones)
        self._update_goals_from_state(state)
        
        if not self.goals_from_yolo:
            return False, ""  # No goals yet, keep running
        
        car_position = np.array(state['car']['position'])
        
        # Check if car is close enough to any YOLO goal
        for goal_x, goal_y, goal_angle in self.goals_from_yolo:
            goal_position = np.array([goal_x, goal_y])
            distance = np.linalg.norm(car_position - goal_position)
            
            if distance <= self.goal_radius:
                return True, f"Reached YOLO goal at ({goal_x:.1f}, {goal_y:.1f})"
        
        return False, ""

    def _update_goals_from_state(self, state):
        """Update goals from current state in case YOLO detected new ones"""
        if state and 'environment' in state:
            for module_state in state['environment']:
                if module_state.get('name') == 'YOLOGoals':
                    new_goals = module_state.get('goals', [])
                    if len(new_goals) != len(self.goals_from_yolo):
                        self.goals_from_yolo = new_goals
                        print(f"YOLOGoalStop: Updated to {len(new_goals)} YOLO goals")
                    break

    def render(self, screen, transform_matrix):
        # Render the YOLO goals as circles
        for goal_x, goal_y, goal_angle in self.goals_from_yolo:
            # Transform goal position to screen coordinates
            goal_screen = transform_matrix @ np.array([goal_x, goal_y, 1])
            goal_screen_pos = (int(goal_screen[0]), int(goal_screen[1]))
            
            # Calculate radius in screen coordinates
            radius_world = self.goal_radius
            radius_screen = int(radius_world * transform_matrix[0, 0])  # Use x-scale
            
            # Draw goal circle (green with red border)
            pygame.draw.circle(screen, (0, 255, 0), goal_screen_pos, radius_screen, 3)
            pygame.draw.circle(screen, (255, 0, 0), goal_screen_pos, radius_screen, 2)

    def get_digest(self):
        return f"YOLOGoalStop(goal_radius={self.goal_radius}, goals_count={len(self.goals_from_yolo)})"

    def get_unified_state(self):
        return {
            'name': 'YOLOGoalStop',
            'goals': self.goals_from_yolo,  # Provide goals for the simulation
            'goal_radius': self.goal_radius
        }