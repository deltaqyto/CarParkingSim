import random
from math import pi, cos, sin, sqrt
import numpy as np
import pygame
from os import cpu_count
import colorsys
from random import uniform


from modules.generic_modules import GenericEnvironment, GenericReward, GenericStop, GenericObservation
from Objects.obstacles import RectObstacle
from Objects.car import render_car

from Simulation.training_schedule import GenericTrainingSchedule
from Simulation.environments import load_env, get_basic_env

from modules.environment_modules import Borders
from modules.reward_functions import GoalEndReward, TimePenalty, CollisionPenalty, DistanceReward
from modules.stop_conditions import StepLimit, CollisionStop
from modules.observation_modules import ClassicalObservation, VisionRaycastObservation

from modules.delta_custom import ObstacleProximityReward


class EnvironmentGoal(GenericEnvironment):
    def __init__(self, angle_tolerance, bidirectional=False, region=None, goal_size=2.0, angle_range=(-pi, pi), goal_distance_range=(5, 100)):
        self.goal_position = [0, 0]
        self.goal_angle = 0
        self.angle_tolerance = angle_tolerance
        self.bidirectional = bidirectional
        self.goal_size = goal_size
        self.goal_distance_range = goal_distance_range

        self.region = region if region is not None else [-30, 30, -20, 20]
        self.angle_range = angle_range

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
        return (f"EnvironmentalGoal(angle_tolerance={round(self.angle_tolerance, 2)}, bidirectional={self.bidirectional}, region={self.region}, "
                f"angle_range={[round(x, 2) for x in self.angle_range]}, goal_size={self.goal_size}, "
                f"goal_distance_range={self.goal_distance_range})")

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
            # if self.bidirectional:
            #     goal_direction = np.array([cos(self.goal_angle), sin(self.goal_angle)])
            #     opposite_end = self.goal_position - goal_direction * direction_length
            #     opposite_end_screen = transform @ np.append(opposite_end, 1)
            #
            #     pygame.draw.line(
            #         screen,
            #         (255, 255, 0),  # Yellow color
            #         (int(goal_position_screen[0]), int(goal_position_screen[1])),
            #         (int(opposite_end_screen[0]), int(opposite_end_screen[1])),
            #         3  # Line width
            #     )

    def get_unified_state(self):
        return {
            'name': 'goal_module',
            'goals': [[*self.goal_position, self.goal_angle]],
            'bidirectional': self.bidirectional,
            'goal_size': self.goal_size,
            'angle_tolerance': self.angle_tolerance
        }


class GoalStop(GenericStop):
    def __init__(self):
        super().__init__()

    def get_digest(self):
        return "GoalStop()"

    def get_unified_state(self):
        return {}

    def check_stop(self, state):
        position = state['car']['position']
        angle = state['car']['angle']

        # Find all goal modules in the environment
        for env_module in state['environment']:
            if env_module.get('name') == 'goal_module':
                # Crash with clear messages if required data is missing
                if 'goals' not in env_module:
                    raise KeyError(f"GoalStop: goal_module missing required 'goals' key. Available keys: {list(env_module.keys())}")

                if 'goal_size' not in env_module:
                    raise KeyError(f"GoalStop: goal_module missing required 'goal_size' key. Available keys: {list(env_module.keys())}")

                if 'angle_tolerance' not in env_module:
                    raise KeyError(f"GoalStop: goal_module missing required 'angle_tolerance' key. Available keys: {list(env_module.keys())}")

                if 'bidirectional' not in env_module:
                    raise KeyError(f"GoalStop: goal_module missing required 'bidirectional' key. Available keys: {list(env_module.keys())}")

                goals = env_module['goals']
                goal_size = env_module['goal_size']
                angle_tolerance = env_module['angle_tolerance']
                bidirectional = env_module['bidirectional']

                # Check each goal in this module
                for i, goal in enumerate(goals):
                    if len(goal) < 3:
                        raise ValueError(f"GoalStop: goal[{i}] has insufficient data. Expected [x, y, angle], got {goal}")

                    goal_position = goal[:2]  # [x, y]
                    goal_angle = goal[2]  # angle

                    # Check distance to goal
                    dist = sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)
                    if dist > goal_size:
                        continue  # Too far from this goal, check next one

                    # Check angle alignment
                    angle_diff = min((angle - goal_angle) % (2 * pi), (goal_angle - angle) % (2 * pi))

                    # If bidirectional, also check the opposite direction
                    if bidirectional:
                        opposite_goal = (goal_angle + pi) % (2 * pi)
                        opposite_diff = min((angle - opposite_goal) % (2 * pi), (opposite_goal - angle) % (2 * pi))
                        angle_diff = min(angle_diff, opposite_diff)

                    # Check if within angle tolerance
                    if angle_diff < angle_tolerance:
                        return True, 'Goal Hit'

        return False, 'Nothing'


class CurbEnvironment(GenericEnvironment):
    def __init__(self, curb_thickness=0.5, curb_gap=10.0, curb_length=20.0, spawn_near=True):
        super().__init__()
        self.curb_thickness = curb_thickness
        self.curb_gap = curb_gap
        self.curb_length = curb_length
        self.spawn_near = spawn_near
        self.collision_rects = []

    def reset(self, mode, state):
        if mode != 'environment':
            return

        self.collision_rects = []

        # Find blocker module data
        goals = []
        for env_module in state['environment']:
            if env_module.get('name') == 'goal_module':
                goals.append(env_module)
                break

        # Get car spawn position
        car_position = state['car']['position']

        for goal in goals:
            goal_size = goal['goal_size']
            for subgoal in goal['goals']:
                goal_position = subgoal[:2]
                goal_angle = subgoal[2]
                self.create_curb_for_goal(
                    goal_position, goal_angle, goal_size, car_position
                )

    def create_curb_for_goal(self, goal_position, goal_angle, goal_size, car_position):

        # Calculate distance from goal center to curb center
        distance = goal_size + self.curb_thickness / 2 + self.curb_gap

        # Goal direction vector
        direction = np.array([np.cos(goal_angle), np.sin(goal_angle)])

        # Two possible curb positions: ahead and behind goal
        pos_ahead = goal_position + distance * direction
        pos_behind = goal_position - distance * direction

        # Choose position closer to car
        dist_to_ahead = np.linalg.norm(pos_ahead - car_position)
        dist_to_behind = np.linalg.norm(pos_behind - car_position)

        if self.spawn_near:
            curb_center = pos_ahead if dist_to_ahead < dist_to_behind else pos_behind
        else:
            curb_center = pos_behind if dist_to_ahead < dist_to_behind else pos_ahead

        # Create curb obstacle
        curb = RectObstacle(
            curb_center,
            [self.curb_length, self.curb_thickness],
            angle=180 / pi * (goal_angle + pi / 2)
        )

        self.collision_rects.append(curb)

    def render(self, screen, transform):
        for curb in self.collision_rects:
            curb.draw(screen, transform)


    def get_digest(self):
        return f"CurbEnvironment(curb_thickness={self.curb_thickness}, curb_gap={self.curb_gap}, curb_length={self.curb_length}, spawn_near={self.spawn_near})"

    def get_unified_state(self):
        return {
            'name': 'curb_module',
            'obstacles': self.collision_rects,
            'curb_thickness': self.curb_thickness,
            'curb_gap': self.curb_gap
        }


class BlockerEnvironment(GenericEnvironment):
    def __init__(self, blockers_per_side=2, blocker_width=2, blocker_length=4.7, blocker_distance=0.2):
        super().__init__()
        self.blockers_per_side = blockers_per_side
        self.blocker_width = blocker_width
        self.blocker_length = blocker_length
        self.blocker_distance = blocker_distance
        self.collision_rects = []
        self.goals_data = []

        self.blocker_colors = []
        for i in range(blockers_per_side * 2):
            hue = uniform(30, 330) / 360.0
            saturation = uniform(0.6, 1.0)
            value = uniform(0.6, 1.0)

            # Convert HSV to RGB and scale to 0-255
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(c * 255) for c in rgb_float)
            self.blocker_colors.append(color)

    def reset(self, mode, state):
        if mode != 'environment':
            return

        self.collision_rects = []
        self.goals_data = []  # Track goals and their blocker info

        # Find all goal modules in the environment
        for env_module in state['environment']:
            if env_module.get('name') == 'goal_module':
                goals = env_module.get('goals', [])
                goal_size = env_module.get('goal_size', 2)

                # Create blockers for each goal
                for goal in goals:
                    goal_position = goal[:2]  # [x, y]
                    goal_angle = goal[2]  # angle

                    # Store goal data for curb generation
                    self.goals_data.append({
                        'position': goal_position,
                        'angle': goal_angle,
                        'size': goal_size
                    })

                    self.create_blockers_for_goal(goal_position, goal_angle, goal_size, state)

        self.blocker_colors = []
        for i in range(self.blockers_per_side * 2):
            hue = uniform(30, 330) / 360.0
            saturation = uniform(0.6, 1.0)
            value = uniform(0.6, 1.0)

            # Convert HSV to RGB and scale to 0-255
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(c * 255) for c in rgb_float)
            self.blocker_colors.append(color)

    def create_blockers_for_goal(self, goal_position, goal_angle, goal_size, state):
        # Calculate perpendicular direction to goal angle
        perp_angle = goal_angle + pi / 2
        perp_vector = np.array([cos(perp_angle), sin(perp_angle)])

        # Create blockers on both sides
        for side in [-1, 1]:  # -1 for one side, +1 for the other
            for i in range(self.blockers_per_side):
                # Calculate distance from goal center for this blocker
                # Edge-to-edge gap = blocker_distance
                # First blocker: goal_size + blocker_distance + blocker_width/2
                # Subsequent blockers: add (blocker_width + blocker_distance) for each
                base_distance = goal_size + self.blocker_distance + self.blocker_width / 2
                additional_distance = i * (self.blocker_width + self.blocker_distance)
                total_distance = base_distance + additional_distance

                # Calculate blocker center position
                blocker_center = goal_position + side * perp_vector * total_distance

                # Create blocker with same rotation as original working code
                blocker = RectObstacle(
                    blocker_center,
                    [self.blocker_width, self.blocker_length],
                    angle=180 / pi * goal_angle - 90  # Convert radians to degrees
                )

                self.collision_rects.append(blocker)

    def render(self, screen, transform):
        # Render all blockers
        for i, blocker in enumerate(self.collision_rects):
            #blocker.draw(screen, transform)
            #return
            width, length = blocker.size
            render_car(screen, transform, blocker.position, car_angle=180 - blocker.angle - 90, width=width, length=length, color=self.blocker_colors[i])

    def get_digest(self):
        return (f"BlockerEnvironment(blockers_per_side={self.blockers_per_side}, "
                f"blocker_width={self.blocker_width}, blocker_length={self.blocker_length}, "
                f"blocker_distance={self.blocker_distance})")

    def get_unified_state(self):
        return {
            'name': 'blocker_module',
            'obstacles': self.collision_rects,
            'blockers_per_side': self.blockers_per_side,
            'blocker_width': self.blocker_width,
            'blocker_length': self.blocker_length,
            'blocker_distance': self.blocker_distance,
            'goals_with_blockers': self._get_goals_with_blockers()
        }

    def _get_goals_with_blockers(self):
        # This will be populated during reset to track which goals have blockers
        return getattr(self, '_goals_data', [])


def parking_env(render=False, goal_size=1.0, angle_tolerance=1.0, blocker_distance=0.2, blocker_width=2, blocker_length=4.7, blockers_per_side=1, episodes=200, eval_line=False, curb_distance=10, vision=False, vision_model=None, enable_curbs=True, rays=12, ray_distance=10):
    if vision_model is not None:
        vision = True

    world_width = 60
    world_aspect = 3 / 4
    world_height = world_width * world_aspect

    eg = EnvironmentGoal(region=[-world_width / 2 * 0.8, world_width / 2 * 0.8, -world_height / 2 * 0.8, world_height / 2 * 0.8],
                     goal_size=goal_size, angle_tolerance=angle_tolerance, bidirectional=True)
    if eval_line:
        eg = EnvironmentGoal(region=[-world_width / 2 * 0.8, world_width / 2 * 0.8, 7, 7], angle_range=(pi/2, pi/2),
                             goal_size=goal_size, angle_tolerance=angle_tolerance, bidirectional=True)

    environment_modules = [Borders(), eg,
                           BlockerEnvironment(blocker_distance=blocker_distance, blockers_per_side=blockers_per_side, blocker_length=blocker_length, blocker_width=blocker_width),
                           ]
    if enable_curbs:
        environment_modules.append(CurbEnvironment(curb_gap=2.5, spawn_near=False))
        environment_modules.append(CurbEnvironment(curb_gap=curb_distance))

    stop_conditions = [StepLimit(step_limit=episodes),
                       CollisionStop(),
                       GoalStop()
                       ]

    reward_functions = [GoalEndReward(),
                        TimePenalty(),
                        CollisionPenalty(reward=-1),
                        DistanceReward(),
                        ObstacleProximityReward()
                        ]

    observation_module = [ClassicalObservation()]
    #if vision_model is not None:
    #    observation_module.append(VisionRaycastObservation(vision_model, show_image=False))

    env = load_env(render=render, world_width=world_width, world_aspect=world_aspect,
                   stop_conditions=stop_conditions, environment_modules=environment_modules, reward_functions=reward_functions,
                   observation_modules=observation_module, generate_vision=vision,
                   rays=rays, max_ray_distance=ray_distance)
    return env


class ParkingSchedule(GenericTrainingSchedule):
    def __init__(self, render=False, start_from_env=0, vision=False, vision_model=None):
        super().__init__(start_from_env=start_from_env)
        self.environments = [parking_env(render=render, goal_size=1, angle_tolerance=pi/4, blocker_distance=20000, curb_distance=20000, vision=vision, vision_model=vision_model, enable_curbs=False),
                             parking_env(render=render, goal_size=1.5, angle_tolerance=pi/3,  blockers_per_side=1, blocker_distance=2, curb_distance=20000, episodes=400, vision=vision, vision_model=vision_model, enable_curbs=False),
                             parking_env(render=render, goal_size=1, angle_tolerance=pi/8,  blockers_per_side=2, blocker_distance=1.6, curb_distance=20000, episodes=400, vision=vision, vision_model=vision_model, enable_curbs=False),
                             parking_env(render=render, goal_size=1, angle_tolerance=pi/8,  blockers_per_side=2, blocker_distance=1.6, episodes=400, curb_distance=20, vision=vision, vision_model=vision_model)]

        base_params = {
            'num_envs': min(14, max(1, cpu_count())),
            'action_dim': 2,
            'batch_size': 512,
            'total_timesteps': 3_000_000,
            'save_freq': 20000,
            'eval_episodes': 10,
            'seed': 41,
            'exploration_noise': 0.15,
            'start_timesteps': 25000,
            'buffer_size': 5_000_000,
            'learning_rate': 3e-4,
            'net_size': [400, 300],
        }
        self.parameters = [base_params, {'start_timesteps': 0, 'total_timesteps': 2_000_000}]
