import random
from math import pi, cos, sin, sqrt, exp
import numpy as np
import pygame
from os import cpu_count

from modules.generic_modules import GenericEnvironment, GenericReward
from Objects.obstacles import RectObstacle
from modules.generic_modules import GenericStop

from Simulation.training_schedule import GenericTrainingSchedule
from Simulation.environments import load_env, get_basic_env

# Yes, these might show as import errors. It resolves at runtime
from modules.environment_modules import Borders
from modules.reward_functions import GoalEndReward, TimePenalty, CollisionPenalty, DistanceReward
from modules.stop_conditions import bidirectional_goal, StepLimit, CollisionStop


class ObstacleProximityReward(GenericReward):
    def __init__(self, reward_factor=-0.02, danger_threshold=4.7, safety_margin=2.0):
        super().__init__()
        self.reward_factor = reward_factor
        self.danger_threshold = danger_threshold
        self.safety_margin = safety_margin

    def get_digest(self):
        return (f'ObstacleProximityReward(reward_factor={self.reward_factor}, '
                f'danger_threshold={self.danger_threshold}, '
                f'safety_margin={self.safety_margin})')

    def get_reward(self, state):
        if 'raycasts_true' not in state or not state['raycasts_true']:
            return 'ObstacleProximityReward', 0

        obstacle_reward = 0
        for distance in state['raycasts_true']:
            if distance < self.danger_threshold + self.safety_margin:
                # Sigmoid function that creates smooth penalty as distance approaches danger_threshold
                proximity_factor = 1 / (1 + exp((distance - self.danger_threshold) * 2))
                obstacle_reward += proximity_factor

        # Apply reward factor to scale the total penalty
        final_reward = self.reward_factor * obstacle_reward

        return 'ObstacleProximityReward', final_reward


class BlockedGoal(GenericEnvironment, GenericStop):
    def __init__(self, angle_tolerance=0.1 * pi, bidirectional=True, region=None, blockers=True, goal_size=2, angle_range=(-pi, pi),
                 goal_distance_range=(10, 100), blocker_width=2, blocker_length=4.7, blocker_distance=2, obstacles=0):
        GenericEnvironment.__init__(self)
        GenericStop.__init__(self)

        # Goal properties
        self.goal_position = [0, 0]
        self.goal_angle = 0
        self.angle_tolerance = angle_tolerance
        self.bidirectional = bidirectional
        self.goal_size = goal_size
        self.goal_distance_range = goal_distance_range
        self.region = region if region is not None else [-30, 30, -20, 20]
        self.angle_range = angle_range

        # Rectangle properties
        self.blockers = blockers
        self.blocker_width = blocker_width
        self.blocker_length = blocker_length
        self.blocker_distance = blocker_distance
        self.collision_rects = []

        # Number of additional random obstacles
        self.obstacles = obstacles

        # For environment tracking
        self.world_width = None
        self.world_height = None

    def reset(self, mode, state=None):
        if mode != 'environment':
            return
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]

        self.pick_goal()
        dist = sqrt((state['car']['position'][0] - self.goal_position[0]) ** 2 +
                    (state['car']['position'][1] - self.goal_position[1]) ** 2)
        while dist < self.goal_distance_range[0] or dist > self.goal_distance_range[1]:
            self.pick_goal()
            dist = sqrt((state['car']['position'][0] - self.goal_position[0]) ** 2 +
                        (state['car']['position'][1] - self.goal_position[1]) ** 2)

        self.goal_angle = random.uniform(*self.angle_range)

        # Create the rectangle obstacles parallel to the goal direction
        self.collision_rects = []
        if self.blockers:
            self.create_rectangle_obstacles()

        # Create additional random obstacles
        if self.obstacles > 0:
            self.create_random_obstacles(state['car']['position'])

    def pick_goal(self):
        self.goal_position = [random.uniform(self.region[0], self.region[1]),
                              random.uniform(self.region[2], self.region[3])]

    def create_rectangle_obstacles(self):
        # Calculate perpendicular direction to goal angle
        perp_angle = self.goal_angle + pi / 2
        perp_vector = np.array([cos(perp_angle), sin(perp_angle)])

        # Calculate positions for rectangles (perpendicular to goal direction)
        rect1_center = self.goal_position + perp_vector * (self.blocker_distance + self.goal_size)
        rect2_center = self.goal_position - perp_vector * (self.blocker_distance + self.goal_size)

        # Create rotated rectangles
        rect1 = RectObstacle(
            rect1_center,
            [self.blocker_width, self.blocker_length],
            angle=180 / pi * self.goal_angle
        )

        rect2 = RectObstacle(
            rect2_center,
            [self.blocker_width, self.blocker_length],
            angle=180 / pi * self.goal_angle
        )

        self.collision_rects = [rect1, rect2]

    def create_random_obstacles(self, car_position):
        """Generate random obstacles within the region"""
        for _ in range(self.obstacles):
            valid_position = False
            pos = [200, 200]
            angle = 0
            while not valid_position:
                # Generate random position within region
                pos = [random.uniform(self.region[0], self.region[1]),
                       random.uniform(self.region[2], self.region[3])]

                # Generate random angle
                angle = random.uniform(-pi, pi)

                # Check distance to goal
                dist_to_goal = sqrt((pos[0] - self.goal_position[0]) ** 2 +
                                    (pos[1] - self.goal_position[1]) ** 2)

                # Check distance to car
                dist_to_car = sqrt((pos[0] - car_position[0]) ** 2 +
                                   (pos[1] - car_position[1]) ** 2)

                # Ensure the obstacle is not too close to the goal or car
                min_distance = self.blocker_distance + self.goal_size

                if dist_to_goal > min_distance and dist_to_car > min_distance:
                    valid_position = True

            # Create a random obstacle with random size
            width = random.uniform(1, self.blocker_width * 1.5)
            length = random.uniform(1, self.blocker_length * 1.5)

            obstacle = RectObstacle(pos, [width, length], angle=180 / pi * angle + 90)

            # Add to collision rectangles list
            self.collision_rects.append(obstacle)

    def check_stop(self, state):
        position = state['car']['position']
        angle = state['car']['angle']

        if sqrt((position[0] - self.goal_position[0]) ** 2 + (position[1] - self.goal_position[1]) ** 2) > self.goal_size:
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
        # Render goal
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
                opposite_end = self.goal_position - goal_direction * direction_length
                opposite_end_screen = transform @ np.append(opposite_end, 1)

                pygame.draw.line(
                    screen,
                    (255, 255, 0),  # Yellow color
                    (int(goal_position_screen[0]), int(goal_position_screen[1])),
                    (int(opposite_end_screen[0]), int(opposite_end_screen[1])),
                    3  # Line width
                )

        # Render rectangle obstacles
        for rect in self.collision_rects:
            rect.draw(screen, transform)

    def get_digest(self):
        return (f"BlockedGoal(angle_tolerance={round(self.angle_tolerance, 2)}, bidirectional={self.bidirectional}, "
                f"region={self.region}, angle_range={[round(x, 2) for x in self.angle_range]}, "
                f"goal_size={self.goal_size}, goal_distance_range={self.goal_distance_range}, "
                f"blocker_width={self.blocker_width}, blocker_length={self.blocker_length}, "
                f"blocker_distance={self.blocker_distance}, obstacles={self.obstacles}, blockers={self.blockers})")

    def get_unified_state(self):
        return {
            'goals': [[*self.goal_position, self.goal_angle]],
            'bidirectional': self.bidirectional,
            'goal_size': self.goal_size,
            'obstacles': self.collision_rects
        }


def delta_env(render=False, goal_size=1, angle_tolerance=1, blocker_distance=4, blocker_length=2, blocker_width=4.7, episodes=200, obstacles=0, blockers=True):
    world_width = 60
    world_aspect = 3 / 4
    world_height = world_width * world_aspect

    bg = BlockedGoal(region=[-world_width / 2 * 0.8, world_width / 2 * 0.8, -world_height / 2 * 0.8, world_height / 2 * 0.8],
                     goal_size=goal_size, angle_tolerance=angle_tolerance, blockers=blockers, blocker_distance=blocker_distance, blocker_length=blocker_length, blocker_width=blocker_width, obstacles=obstacles)
    environment_modules = [Borders(), bg]

    stop_conditions = [StepLimit(step_limit=episodes),
                       CollisionStop(),
                       bg]

    reward_functions = [GoalEndReward(),
                        TimePenalty(),
                        CollisionPenalty(reward=-1),
                        DistanceReward(),
                        ObstacleProximityReward(),
                        ]

    env = load_env(render=render, world_width=world_width, world_aspect=world_aspect,
                   stop_conditions=stop_conditions, environment_modules=environment_modules, reward_functions=reward_functions)
    return env



class DeltaTrainSchedule(GenericTrainingSchedule):
    def __init__(self, render=False):
        super().__init__()
        self.environments = [delta_env(render=render, goal_size=2,   angle_tolerance=pi/4,  blockers=False, obstacles=2),
                             delta_env(render=render, goal_size=1,   angle_tolerance=pi/8,  blockers=False, obstacles=5),
                             #delta_env(render=render, goal_size=0.5, angle_tolerance=pi/10, blockers=False, obstacles=2),  # Remove for next run, too hard
                             delta_env(render=render, goal_size=1.5, angle_tolerance=pi/3,  blocker_length=2, blocker_distance=4, episodes=400, obstacles=4),
                             delta_env(render=render, goal_size=1.5,   angle_tolerance=pi/4,  blocker_length=3, blocker_width=3, blocker_distance=3, episodes=400, obstacles=0),
                             delta_env(render=render, goal_size=1,   angle_tolerance=pi/8,  blocker_length=3, blocker_width=4.7, blocker_distance=2.5, episodes=400, obstacles=6)]

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
        self.parameters = [base_params, {'total_timesteps': 1_500_000}, {'total_timesteps': 2_000_000}]
