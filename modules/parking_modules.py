from random import random, choice, sample
from math import pi, cos, sin, sqrt, exp
import numpy as np
import pygame
from os import cpu_count, path
import colorsys

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
from modules.contingency_custom import GoalStop, EnvironmentGoal
from modules.module_reward_display import RewardDisplayModule

from modules.contingency_custom import parking_env


class EnvironmentSelector(GenericEnvironment):
    def __init__(self, environment_type="straight_lot", debug=False):
        super().__init__()
        self.environment_type = environment_type
        self.debug = debug
        self.state_data = {}

    def reset(self, mode, state):
        if mode != 'environment':
            return

        self.state_data = {
            'name': 'environment_selector_module',
            'environment': self.environment_type,

        }

    def render(self, screen, transform):
        if not self.debug:
            return

        font = pygame.font.Font(None, 36)
        text = font.render(f"EnvSelector: {self.environment_type}", True, (255, 255, 0))
        screen.blit(text, (10, 10))

    def get_digest(self):
        return f"EnvironmentSelector(environment_type={self.environment_type}, debug={self.debug})"

    def get_unified_state(self):
        return self.state_data


class LotEnvironmentPlanner(GenericEnvironment):
    def __init__(self, main_road_width=15, parking_row_depth=5, margin=0,
                 curb_enabled=False, curb_depth=1.0, curb_thickness=1.0, debug=False):
        super().__init__()
        self.main_road_width = main_road_width
        self.parking_row_depth = parking_row_depth
        self.margin = margin
        self.curb_enabled = curb_enabled
        self.curb_depth = curb_depth
        self.curb_thickness = curb_thickness
        self.debug = debug
        self.effective_world_x = None
        self.effective_world_y = None

        self.spawn_points = []
        self.waypoint_goal = None
        self.lot_orientation = None
        self.world_bounds = None
        self.environment_type = None

    def reset(self, mode, state):
        if mode != 'environment':
            return

        found_selector = False
        self.environment_type = None
        for env_module in state['environment']:
            if env_module.get('name') == 'environment_selector_module':
                self.environment_type = env_module.get('environment')
                if self.environment_type not in ['straight_lot', 'branching_lot']:
                    return
                found_selector = True
                break

        if not found_selector:
            raise KeyError("LotEnvironmentPlanner: environment_selector_module not found in state")

        self.world_bounds = state['world_size']
        world_x, world_y = self.world_bounds
        world_x /= 2
        world_y /= 2
        self.world_bounds = (world_x, world_y)

        # Apply margin to reduce effective world size
        self.effective_world_x = world_x - self.margin
        self.effective_world_y = world_y - self.margin

        self.spawn_points = []
        self.lot_orientation = choice(['vertical', 'horizontal'])

        if self.environment_type == 'branching_lot':
            self.lot_orientation = 'vertical'
            self._create_branching_lots(self.effective_world_x, self.effective_world_y)
            self._process_waypoint_eligibility_branching()
        else:  # straight_lot (default)
            self._create_parking_strips(self.effective_world_x, self.effective_world_y)
            if self.curb_enabled:
                self._create_curbs(self.effective_world_x, self.effective_world_y)

        self._create_main_road(self.effective_world_x, self.effective_world_y)

    def _create_branching_lots(self, world_x, world_y):
        """Create perpendicular parking lots branching off the main road with configurable spacing"""
        start_y = -world_y + self.parking_row_depth / 2 + self.main_road_width

        i = 0
        while True:
            branch_y = start_y + i * (self.parking_row_depth * 2 + self.main_road_width + self.curb_depth * 2 + self.curb_thickness)
            if abs(branch_y) > world_y:
                break

            branch_depth = world_x - self.main_road_width / 2

            # Left side branch - add waypoint eligibility bool
            left_x = -self.main_road_width / 2
            left_spawn = ('parking_lot', left_x, branch_y, pi, self.parking_row_depth, branch_depth, True)
            self.spawn_points.append(left_spawn)

            right_x = self.main_road_width / 2
            right_spawn = ('parking_lot', right_x, branch_y, 0, self.parking_row_depth, branch_depth, True)
            self.spawn_points.append(right_spawn)

            curb_x = self.main_road_width / 2
            curb_y = branch_y + self.parking_row_depth / 2 + self.curb_depth + self.curb_thickness / 2
            self.spawn_points.append(('curb', curb_x, curb_y, pi / 2, branch_depth, self.curb_thickness))
            curb_x = -self.main_road_width / 2
            curb_y = branch_y + self.parking_row_depth / 2 + self.curb_depth + self.curb_thickness / 2
            self.spawn_points.append(('curb', curb_x, curb_y, -pi / 2, branch_depth, self.curb_thickness))

            if (branch_y + self.parking_row_depth + self.curb_depth * 2 + self.curb_thickness) > world_y:
                break
            low_y = branch_y + self.parking_row_depth + self.curb_depth * 2 + self.curb_thickness
            left_x = -self.main_road_width / 2
            left_spawn = ('parking_lot', left_x, low_y, pi, self.parking_row_depth, branch_depth, True)
            self.spawn_points.append(left_spawn)
            right_x = self.main_road_width / 2
            right_spawn = ('parking_lot', right_x, low_y, 0, self.parking_row_depth, branch_depth, True)
            self.spawn_points.append(right_spawn)

            self.spawn_points.append(('road', -world_x, branch_y - self.parking_row_depth / 2 - self.main_road_width / 2, 0, self.main_road_width, 2 * world_x))

            i += 1
        self.spawn_points.append(('road', -world_x, branch_y - self.parking_row_depth / 2 - self.main_road_width / 2, 0, self.main_road_width, 2 * world_x))

    def _process_waypoint_eligibility_branching(self):
        """For branching lots, randomly select one parking lot to be eligible for waypoint spawning"""
        # Get all parking lot indices
        parking_lot_indices = []
        for i, spawn_point in enumerate(self.spawn_points):
            if spawn_point[0] == 'parking_lot':
                parking_lot_indices.append(i)

        if not parking_lot_indices:
            return

        # Randomly select one parking lot to remain eligible
        selected_index = choice(parking_lot_indices)

        # Set all parking lots to ineligible except the selected one
        for i in parking_lot_indices:
            spawn_point = self.spawn_points[i]
            if i == selected_index:
                # Keep this one eligible and create waypoint goal
                spawn_type, x, y, direction, width, length, _ = spawn_point
                self.spawn_points[i] = (spawn_type, x, y, direction, width, length, True)

                # Find the corresponding road for waypoint goal
                road_x, road_y = self._find_road_for_lot(x, y)
                if road_x is not None and road_y is not None:
                    waypoint_goal = [(-self.main_road_width/2 - 2 if x < 0 else self.main_road_width/2 + 2), road_y, 0]
                    self.waypoint_goal = waypoint_goal
            else:
                # Make this one ineligible
                spawn_type, x, y, direction, width, length, _ = spawn_point
                self.spawn_points[i] = (spawn_type, x, y, direction, width, length, False)

    def _find_road_for_lot(self, lot_x, lot_y):
        """Find the road spawn point that corresponds to this parking lot"""
        # For branching lots, find the road at the same y level
        best_road = 0, 0
        best_road_dist = 1e6
        for i, spawn_point in enumerate(self.spawn_points):
            if spawn_point[0] == 'road':
                road_type, road_x, road_y, road_direction, road_width, road_length = spawn_point
                # Check if this road is at approximately the same y level as the lot
                if abs(road_y - lot_y) < best_road_dist:
                    best_road = road_x, road_y
                    best_road_dist = abs(road_y - lot_y)
        return best_road

    def _create_main_road(self, world_x, world_y):
        if self.lot_orientation == 'vertical':
            # Vertical main road at x=0
            main_road_spawn = ('road', 0, -world_y, pi / 2, self.main_road_width, 2 * world_y)
        else:
            # Horizontal main road at y=0
            main_road_spawn = ('road', -world_x, 0, 0, self.main_road_width, 2 * world_x)

        self.spawn_points.append(main_road_spawn)

    def _create_parking_strips(self, world_x, world_y):
        if self.lot_orientation == 'vertical':
            self._create_vertical_parking_strips(world_x, world_y)
        else:
            self._create_horizontal_parking_strips(world_x, world_y)

    def _create_vertical_parking_strips(self, world_x, world_y):
        # Right side parking strip - add waypoint eligibility bool
        right_x = self.main_road_width / 2 + self.parking_row_depth / 2
        right_spawn = ('parking_lot', right_x, -world_y, pi / 2, self.parking_row_depth, 2 * world_y, True)
        self.spawn_points.append(right_spawn)

        # Left side parking strip
        left_x = -self.main_road_width / 2 - self.parking_row_depth / 2
        left_spawn = ('parking_lot', left_x, -world_y, pi / 2, self.parking_row_depth, 2 * world_y, True)
        self.spawn_points.append(left_spawn)

    def _create_horizontal_parking_strips(self, world_x, world_y):
        # Top parking strip - add waypoint eligibility bool
        top_y = self.main_road_width / 2 + self.parking_row_depth / 2
        top_spawn = ('parking_lot', -world_x, top_y, 0, self.parking_row_depth, 2 * world_x, True)
        self.spawn_points.append(top_spawn)

        # Bottom parking strip
        bottom_y = -self.main_road_width / 2 - self.parking_row_depth / 2
        bottom_spawn = ('parking_lot', -world_x, bottom_y, 0, self.parking_row_depth, 2 * world_x, True)
        self.spawn_points.append(bottom_spawn)

    def _create_curbs(self, world_x, world_y):
        # Create curbs for each parking lot
        parking_lots = [sp for sp in self.spawn_points if sp[0] == 'parking_lot']

        for spawn_data in parking_lots:
            spawn_type, lot_x, lot_y, lot_direction, lot_width, lot_length = spawn_data[:6]  # Ignore waypoint bool

            if self.lot_orientation == 'vertical':
                # Lots are on left/right edges, curbs need to move further out in x
                curb_x = lot_x + (lot_width / 2 + self.curb_depth + self.curb_thickness / 2) * (1 if lot_x > 0 else -1)
                curb_y = lot_y
                curb_spawn = ('curb', curb_x, curb_y, 0, 2 * world_y, self.curb_thickness)
            else:  # horizontal
                # Lots are on top/bottom edges, curbs need to move further out in y
                curb_x = lot_x
                curb_y = lot_y + (lot_width / 2 + self.curb_depth + self.curb_thickness / 2) * (1 if lot_y > 0 else -1)
                curb_spawn = ('curb', curb_x, curb_y, pi / 2, 2 * world_x, self.curb_thickness)

            self.spawn_points.append(curb_spawn)

    def render(self, screen, transform):
        # Render waypoint goals if any
        if self.waypoint_goal is not None:
            waypoint_x, waypoint_y, waypoint_angle = self.waypoint_goal
            # Draw goal circle
            goal_position_screen = transform @ np.array([waypoint_x, waypoint_y, 1])

            # Calculate scaled radius for goal circle
            origin_point = [0, 0]
            radius_point = [2, 0]
            origin_screen = transform @ np.array([*origin_point, 1])
            radius_screen = transform @ np.array([*radius_point, 1])
            dx = radius_screen[0] - origin_screen[0]
            dy = radius_screen[1] - origin_screen[1]
            scaled_radius = int(np.sqrt(dx ** 2 + dy ** 2))

            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (int(goal_position_screen[0]), int(goal_position_screen[1])),
                scaled_radius,
                0
            )

            direction_length = 2
            goal_direction = np.array([cos(waypoint_angle), sin(waypoint_angle)])
            direction_end = [waypoint_x, waypoint_y] + goal_direction * direction_length
            direction_end_screen = transform @ np.array([*direction_end, 1])

            pygame.draw.line(
                screen,
                (255, 255, 0),
                (int(goal_position_screen[0]), int(goal_position_screen[1])),
                (int(direction_end_screen[0]), int(direction_end_screen[1])),
                3
            )

        if not self.debug:
                return

        for spawn_point in self.spawn_points:
            spawn_type = spawn_point[0]
            x, y, direction, width, length = spawn_point[1:6]

            cos_dir = cos(direction)
            sin_dir = sin(direction)
            half_width = width / 2

            corners = [
                [0, -half_width],
                [length, -half_width],
                [length, half_width],
                [0, half_width]
            ]

            world_corners = []
            for corner_x, corner_y in corners:
                rotated_x = corner_x * cos_dir - corner_y * sin_dir + x
                rotated_y = corner_x * sin_dir + corner_y * cos_dir + y
                world_corners.append([rotated_x, rotated_y, 1])

            screen_corners = []
            for corner in world_corners:
                screen_corner = transform @ np.array(corner)
                screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

            color = {
                'road': (100, 100, 100),
                'parking_lot': (200, 200, 200),
                'curb': (128, 128, 128)
            }.get(spawn_type, (255, 0, 255))

            pygame.draw.polygon(screen, color, screen_corners, 2)

            center_screen = transform @ np.array([x, y, 1])
            pygame.draw.circle(screen, (255, 255, 0),
                               (int(center_screen[0]), int(center_screen[1])), 3)


    def get_digest(self):
        return (f"LotEnvironmentPlanner(main_road_width={self.main_road_width}, "
                f"parking_row_depth={self.parking_row_depth}, margin={self.margin}, "
                f"curb_enabled={self.curb_enabled}, curb_depth={self.curb_depth}, "
                f"curb_thickness={self.curb_thickness}, debug={self.debug})")

    def get_unified_state(self):
        if self.environment_type == "branching_lot":
            return {
                'name': 'lot_planner_module',
                'spawn_points': self.spawn_points,
                'lot_orientation': self.lot_orientation,
                'car_orientation': -90,
                'car_position': [0, self.effective_world_y - 15],
                'world_bounds': self.world_bounds,
                'effective_world_bounds': [self.effective_world_x, self.effective_world_y],
                'curb_enabled': self.curb_enabled,
                'goals': [self.waypoint_goal] if self.waypoint_goal is not None else [],
            }
        return {
            'name': 'lot_planner_module',
            'spawn_points': self.spawn_points,
            'lot_orientation': self.lot_orientation,
            'car_orientation': 0 if self.lot_orientation == 'horizontal' else 90,
            'world_bounds': self.world_bounds,
            'effective_world_bounds': [self.effective_world_x, self.effective_world_y],
            'curb_enabled': self.curb_enabled,
        }


class ParkedCarRow(GenericEnvironment):
    def __init__(self, car_width=2, car_length=4.7, parking_spot_width=3,
                 vacancy_rate=0.1, min_missing=1, debug=False, force_first_car=False):
        super().__init__()
        self.car_width = car_width
        self.car_length = car_length
        self.parking_spot_width = parking_spot_width
        self.force_first_car = force_first_car
        self.vacancy_rate = vacancy_rate
        self.min_missing = min_missing
        self.debug = debug

        self.parked_cars = []  # Clustered collision rectangles
        self.individual_cars = []  # Individual car data for rendering
        self.missing_cars = []
        self.parking_lot_spawns = []
        self.all_spots = []

        self.car_colors = []

    def reset(self, mode, state):
        if mode != 'environment':
            return

        found_planner = False
        for env_module in state['environment']:
            if env_module.get('name') == 'lot_planner_module':
                found_planner = True
                spawn_points = env_module.get('spawn_points', [])
                # Filter for parking lots that can spawn waypoints/goals
                self.parking_lot_spawns = []
                for sp in spawn_points:
                    if sp[0] == 'parking_lot':
                        can_spawn_waypoint = sp[6] if len(sp) > 6 else True
                        self.parking_lot_spawns.append((sp, can_spawn_waypoint))
                break

        if not found_planner:
            raise KeyError("ParkedCarRow: lot_planner_module not found in state")

        self.parked_cars = []
        self.individual_cars = []
        self.missing_cars = []
        self.car_colors = []
        self.all_spots = []

        lots_to_process = self.parking_lot_spawns[:1] if self.debug else self.parking_lot_spawns

        for spawn_data in lots_to_process:
            spawn_data, can_spawn_waypoint = spawn_data
            spawn_type, x, y, direction, width, length = spawn_data[:6]
            self._create_parking_row(x, y, direction, width, length, can_spawn_waypoint)

    def _create_parking_row(self, lot_x, lot_y, lot_direction, lot_width, lot_length, can_spawn_waypoint):
        num_spots = int(lot_length / self.parking_spot_width)
        if num_spots == 0:
            return

        if num_spots == 1:
            spot_spacing = 0
            start_offset = lot_length / 2
        else:
            spot_spacing = lot_length / num_spots
            start_offset = spot_spacing / 2

        cos_dir = cos(lot_direction)
        sin_dir = sin(lot_direction)

        # Generate all spot positions
        all_spots = []
        for i in range(num_spots):
            spot_offset = start_offset + i * spot_spacing
            car_x = lot_x + spot_offset * cos_dir
            car_y = lot_y + spot_offset * sin_dir

            if random() < 0.5:
                car_direction = lot_direction + pi / 2
            else:
                car_direction = lot_direction - pi / 2

            all_spots.append((car_x, car_y, car_direction, i))  # Added index

        self.all_spots.extend(all_spots)

        # Determine parked vs missing spots
        spot_states = []  # True = parked, False = missing
        for _ in range(num_spots):
            spot_states.append(random() >= self.vacancy_rate)
        if self.force_first_car and spot_states:
            spot_states[0] = True

        # Ensure minimum missing cars
        missing_count = spot_states.count(False)
        if missing_count < self.min_missing:
            parked_indices = [i for i, state in enumerate(spot_states) if state]
            to_convert = min(self.min_missing - missing_count, len(parked_indices))

            if to_convert > 0:
                convert_indices = sample(parked_indices, to_convert)
                for idx in convert_indices:
                    spot_states[idx] = False

        # Create missing cars list
        if can_spawn_waypoint:
            for i, (car_x, car_y, car_direction, _) in enumerate(all_spots):
                if not spot_states[i]:
                    self.missing_cars.append((car_x, car_y, car_direction))

        # Cluster adjacent parked cars
        clusters = self._find_car_clusters(spot_states)

        for cluster_indices in clusters:
            if len(cluster_indices) == 1:
                # Single car - create individual obstacle
                idx = cluster_indices[0]
                car_x, car_y, car_direction, _ = all_spots[idx]

                car_obstacle = RectObstacle(
                    [car_x, car_y],
                    [self.car_width, self.car_length],
                    angle=180 / pi * (car_direction - pi / 2)
                )
                car_obstacle.car_direction = car_direction
                car_obstacle.cluster_size = 1
                self.parked_cars.append(car_obstacle)

                # Store individual car for rendering
                self.individual_cars.append((car_x, car_y, car_direction))

            else:
                # Multiple adjacent cars - create cluster obstacle
                cluster_cars = [all_spots[i] for i in cluster_indices]
                cluster_obstacle = self._create_cluster_obstacle(
                    cluster_cars, lot_direction, spot_spacing
                )
                self.parked_cars.append(cluster_obstacle)

                # Store individual cars for rendering
                for car_x, car_y, car_direction, _ in cluster_cars:
                    self.individual_cars.append((car_x, car_y, car_direction))

    def _find_car_clusters(self, spot_states):
        """Find groups of consecutive parked cars."""
        clusters = []
        current_cluster = []

        for i, is_parked in enumerate(spot_states):
            if is_parked:
                current_cluster.append(i)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []

        # Don't forget the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _create_cluster_obstacle(self, cluster_cars, lot_direction, spot_spacing):
        if not cluster_cars:
            return None

        cluster_center_x = sum(car[0] for car in cluster_cars) / len(cluster_cars)
        cluster_center_y = sum(car[1] for car in cluster_cars) / len(cluster_cars)

        car_direction = cluster_cars[0][2]
        cluster_span = self.car_width + (len(cluster_cars) - 1) * spot_spacing

        cluster_obstacle = RectObstacle(
            [cluster_center_x, cluster_center_y],
            [cluster_span, self.car_length],
            angle=180 / pi * (car_direction - pi / 2)
        )
        cluster_obstacle.car_direction = car_direction
        cluster_obstacle.cluster_size = len(cluster_cars)

        return cluster_obstacle

    def render(self, screen, transform):
        # Render parking spot backgrounds
        self._render_parking_spots(screen, transform)

        # Render individual cars (not clusters)
        for i, (car_x, car_y, car_direction) in enumerate(self.individual_cars):
            if i < len(self.car_colors):
                color = self.car_colors[i]
                render_car(screen, transform, [car_x, car_y],
                           car_angle=180 / pi * car_direction,
                           width=self.car_width, length=self.car_length,
                           color=color)
            else:
                color = render_car(screen, transform, [car_x, car_y],
                                   car_angle=180 / pi * car_direction,
                                   width=self.car_width, length=self.car_length)
                self.car_colors.append(color)

        if self.debug:
            # Draw cluster boundaries in blue
            for cluster in self.parked_cars:
                if hasattr(cluster, 'cluster_size') and cluster.cluster_size > 1:
                    self._render_cluster_bounds(screen, transform, cluster)

            # Draw missing car spots as red outlines
            for car_x, car_y, car_direction in self.missing_cars:
                self._render_missing_spot(screen, transform, car_x, car_y, car_direction)

            # Draw parking lot spawn rectangles as green outlines
            for spawn_data in self.parking_lot_spawns:
                spawn_type, x, y, direction, width, length = spawn_data[:6]  # Ignore waypoint bool
                self._render_lot_bounds(screen, transform, x, y, direction, width, length)

                center_screen = transform @ np.array([x, y, 1])
                pygame.draw.circle(screen, (0, 0, 255),
                                   (int(center_screen[0]), int(center_screen[1])), 3)

    def _render_cluster_bounds(self, screen, transform, cluster):
        """Render cluster collision bounds for debugging."""
        cos_dir = cos(cluster.car_direction)
        sin_dir = sin(cluster.car_direction)
        half_width = cluster.size[0] / 2
        half_length = cluster.size[1] / 2

        corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + cluster.position[0]
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + cluster.position[1]
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        # Draw blue outline for cluster bounds
        pygame.draw.polygon(screen, (0, 0, 255), screen_corners, 2)

    def _render_parking_spots(self, screen, transform):
        lots_to_process = self.parking_lot_spawns[:1] if self.debug else self.parking_lot_spawns

        for spawn_data in lots_to_process:
            spawn_type, lot_x, lot_y, lot_direction, lot_width, lot_length = spawn_data[0][:6]  # Ignore waypoint bool
            self._render_row_spots(screen, transform, lot_x, lot_y, lot_direction, lot_width, lot_length)

    def _render_row_spots(self, screen, transform, lot_x, lot_y, lot_direction, lot_width, lot_length):
        num_spots = int(lot_length / self.parking_spot_width)
        if num_spots == 0:
            return

        if num_spots == 1:
            spot_spacing = 0
            start_offset = lot_length / 2
        else:
            spot_spacing = lot_length / num_spots
            start_offset = spot_spacing / 2

        cos_dir = cos(lot_direction)
        sin_dir = sin(lot_direction)
        perp_cos = cos(lot_direction + pi / 2)
        perp_sin = sin(lot_direction + pi / 2)

        # Draw each parking spot
        for i in range(num_spots):
            spot_offset = start_offset + i * spot_spacing
            spot_center_x = lot_x + spot_offset * cos_dir
            spot_center_y = lot_y + spot_offset * sin_dir

            half_spot_length = spot_spacing / 2 if num_spots > 1 else lot_length / 2
            half_spot_width = lot_width / 2

            corners = [
                [-half_spot_length, -half_spot_width],
                [half_spot_length, -half_spot_width],
                [half_spot_length, half_spot_width],
                [-half_spot_length, half_spot_width]
            ]

            world_corners = []
            for corner_x, corner_y in corners:
                rotated_x = corner_x * cos_dir - corner_y * perp_cos + spot_center_x
                rotated_y = corner_x * sin_dir - corner_y * perp_sin + spot_center_y
                world_corners.append([rotated_x, rotated_y, 1])

            screen_corners = []
            for corner in world_corners:
                screen_corner = transform @ np.array(corner)
                screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

            pygame.draw.polygon(screen, (40, 40, 40), screen_corners)

        # Draw yellow separator lines between spots
        for i in range(num_spots - 1):
            line_offset = start_offset + (i + 1) * spot_spacing - spot_spacing / 2
            line_center_x = lot_x + line_offset * cos_dir
            line_center_y = lot_y + line_offset * sin_dir

            half_width = lot_width / 2
            line_start_x = line_center_x - half_width * perp_cos
            line_start_y = line_center_y - half_width * perp_sin
            line_end_x = line_center_x + half_width * perp_cos
            line_end_y = line_center_y + half_width * perp_sin

            start_screen = transform @ np.array([line_start_x, line_start_y, 1])
            end_screen = transform @ np.array([line_end_x, line_end_y, 1])

            pygame.draw.line(screen, (255, 255, 0),
                             (int(start_screen[0]), int(start_screen[1])),
                             (int(end_screen[0]), int(end_screen[1])), 2)

        # Draw light grey lines on front and back edges
        half_width = lot_width / 2

        # Front edge
        front_edge_x = lot_x
        front_edge_y = lot_y
        front_start_x = front_edge_x - half_width * perp_cos
        front_start_y = front_edge_y - half_width * perp_sin
        front_end_x = front_edge_x + half_width * perp_cos
        front_end_y = front_edge_y + half_width * perp_sin

        front_start_screen = transform @ np.array([front_start_x, front_start_y, 1])
        front_end_screen = transform @ np.array([front_end_x, front_end_y, 1])

        pygame.draw.line(screen, (100, 100, 100),
                         (int(front_start_screen[0]), int(front_start_screen[1])),
                         (int(front_end_screen[0]), int(front_end_screen[1])), 2)

        # Back edge
        back_edge_x = lot_x + lot_length * cos_dir
        back_edge_y = lot_y + lot_length * sin_dir
        back_start_x = back_edge_x - half_width * perp_cos
        back_start_y = back_edge_y - half_width * perp_sin
        back_end_x = back_edge_x + half_width * perp_cos
        back_end_y = back_edge_y + half_width * perp_sin

        back_start_screen = transform @ np.array([back_start_x, back_start_y, 1])
        back_end_screen = transform @ np.array([back_end_x, back_end_y, 1])

        pygame.draw.line(screen, (100, 100, 100),
                         (int(back_start_screen[0]), int(back_start_screen[1])),
                         (int(back_end_screen[0]), int(back_end_screen[1])), 2)

    def _render_lot_bounds(self, screen, transform, x, y, direction, width, length):
        cos_dir = cos(direction)
        sin_dir = sin(direction)
        half_width = width / 2

        corners = [
            [0, -half_width],
            [length, -half_width],
            [length, half_width],
            [0, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        pygame.draw.polygon(screen, (0, 255, 0), screen_corners, 2)

    def _render_missing_spot(self, screen, transform, car_x, car_y, car_direction):
        cos_dir = cos(car_direction)
        sin_dir = sin(car_direction)
        half_width = self.car_width / 2
        half_length = self.car_length / 2

        corners = [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + car_x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + car_y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        pygame.draw.polygon(screen, (255, 0, 0), screen_corners, 2)

        center_screen = transform @ np.array([car_x, car_y, 1])
        pygame.draw.circle(screen, (255, 0, 0),
                           (int(center_screen[0]), int(center_screen[1])), 2)

    def get_digest(self):
        return (f"ParkedCarRow(car_width={self.car_width}, car_length={self.car_length}, "
                f"parking_spot_width={self.parking_spot_width}, "
                f"vacancy_rate={self.vacancy_rate}, min_missing={self.min_missing}, "
                f"force_first_car={self.force_first_car}, "
                f"debug={self.debug})")

    def get_unified_state(self):
        return {
            'name': 'parked_car_row_module',
            'obstacles': self.parked_cars,  # Clustered collision rectangles
            'missing_cars': self.missing_cars,
            'total_spots': len(self.individual_cars) + len(self.missing_cars),
            'total_clusters': len(self.parked_cars)
        }

class RoadRenderer(GenericEnvironment):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.road_spawns = []

    def reset(self, mode, state):
        if mode != 'environment':
            return

        found_planner = False
        for env_module in state['environment']:
            if env_module.get('name') == 'lot_planner_module':
                found_planner = True
                spawn_points = env_module.get('spawn_points', [])
                self.road_spawns = [sp for sp in spawn_points if sp[0] == 'road']
                break

        if not found_planner:
            raise KeyError("RoadRenderer: lot_planner_module not found in state")

    def render(self, screen, transform):
        # Draw road surfaces
        for spawn_type, x, y, direction, width, length in self.road_spawns:
            self._render_road_surface(screen, transform, x, y, direction, width, length, (60, 60, 60))

        # Draw dashed midlines
        for spawn_type, x, y, direction, width, length in self.road_spawns:
            self._render_dashed_midline(screen, transform, x, y, direction, width, length)

        # Debug rendering
        if self.debug:
            for spawn_type, x, y, direction, width, length in self.road_spawns:
                self._render_debug_info(screen, transform, x, y, direction, width, length, (255, 0, 0))

    def _render_road_surface(self, screen, transform, x, y, direction, width, length, color):
        cos_dir = cos(direction)
        sin_dir = sin(direction)
        half_width = width / 2

        corners = [
            [0, -half_width],
            [length, -half_width],
            [length, half_width],
            [0, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        pygame.draw.polygon(screen, color, screen_corners)

    def _render_dashed_midline(self, screen, transform, x, y, direction, width, length):
        cos_dir = cos(direction)
        sin_dir = sin(direction)

        dash_length = 2.0
        gap_length = 1.0
        segment_length = dash_length + gap_length
        num_segments = int(length / segment_length)

        for i in range(num_segments + 1):
            dash_start_local = i * segment_length
            dash_end_local = min(dash_start_local + dash_length, length)

            if dash_start_local >= length:
                break

            start_world_x = dash_start_local * cos_dir + x
            start_world_y = dash_start_local * sin_dir + y
            end_world_x = dash_end_local * cos_dir + x
            end_world_y = dash_end_local * sin_dir + y

            start_screen = transform @ np.array([start_world_x, start_world_y, 1])
            end_screen = transform @ np.array([end_world_x, end_world_y, 1])

            pygame.draw.line(screen, (255, 255, 255),
                             (int(start_screen[0]), int(start_screen[1])),
                             (int(end_screen[0]), int(end_screen[1])), 2)

    def _render_debug_info(self, screen, transform, x, y, direction, width, length, color):
        center_screen = transform @ np.array([x, y, 1])
        pygame.draw.circle(screen, color,
                           (int(center_screen[0]), int(center_screen[1])), 5)

        cos_dir = cos(direction)
        sin_dir = sin(direction)
        half_width = width / 2

        corners = [
            [0, -half_width], [length, -half_width],
            [length, half_width], [0, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        pygame.draw.polygon(screen, color, screen_corners, 2)

    def get_digest(self):
        return f"RoadRenderer(debug={self.debug})"

    def get_unified_state(self):
        return {
            'name': 'road_renderer_module',
            'road_count': len(self.road_spawns)
        }


class GoalSelector(GenericEnvironment):
    def __init__(self, goal_size=2.0, angle_tolerance=pi / 4, bidirectional=True,
                 num_goals=1, debug=False):
        super().__init__()
        self.goal_size = goal_size
        self.angle_tolerance = angle_tolerance
        self.bidirectional = bidirectional
        self.num_goals = num_goals
        self.debug = debug
        self.waypoints = []
        self.got_waypoint = False
        self.reached_waypoint = False

        self.selected_goals = [[50, 50, 0]]

    def reset(self, mode, state):
        if mode != 'environment':
            return

        found_parked_cars = False
        missing_cars = []
        self.waypoints = []
        self.got_waypoint = False
        self.reached_waypoint = False

        for env_module in state['environment']:
            if env_module.get('name') == 'parked_car_row_module':
                found_parked_cars = True
                missing_cars = env_module.get('missing_cars', [])
            if env_module.get('name') == 'lot_planner_module':
                self.waypoints = env_module.get('goals', [])
                self.got_waypoint = len(self.waypoints) > 0

        if not found_parked_cars:
            raise KeyError("GoalSelector: parked_car_row_module not found in state")

        if not missing_cars:
            raise ValueError("GoalSelector: no missing cars available for goal selection")

        # Select goals from available missing cars
        num_to_select = min(self.num_goals, len(missing_cars))
        self.selected_goals = sample(missing_cars, num_to_select)

    def render(self, screen, transform):
        for car_x, car_y, car_direction in self.waypoints if self.got_waypoint and not self.reached_waypoint else self.selected_goals:
            # Draw goal circle
            goal_position_screen = transform @ np.array([car_x, car_y, 1])

            # Calculate scaled radius for goal circle
            origin_point = [0, 0]
            radius_point = [self.goal_size, 0]
            origin_screen = transform @ np.array([*origin_point, 1])
            radius_screen = transform @ np.array([*radius_point, 1])
            dx = radius_screen[0] - origin_screen[0]
            dy = radius_screen[1] - origin_screen[1]
            scaled_radius = int(np.sqrt(dx ** 2 + dy ** 2))

            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (int(goal_position_screen[0]), int(goal_position_screen[1])),
                scaled_radius,
                0
            )

            # Draw goal direction indicator if angle tolerance is restrictive
            if self.angle_tolerance < pi:
                direction_length = 2
                goal_direction = np.array([cos(car_direction), sin(car_direction)])
                direction_end = [car_x, car_y] + goal_direction * direction_length
                direction_end_screen = transform @ np.array([*direction_end, 1])

                pygame.draw.line(
                    screen,
                    (255, 255, 0),
                    (int(goal_position_screen[0]), int(goal_position_screen[1])),
                    (int(direction_end_screen[0]), int(direction_end_screen[1])),
                    3
                )

            if self.debug:
                # Draw debug info
                center_screen = transform @ np.array([car_x, car_y, 1])
                pygame.draw.circle(screen, (255, 0, 255),
                                   (int(center_screen[0]), int(center_screen[1])), 2)

    def get_digest(self):
        return (f"GoalSelector(goal_size={self.goal_size}, "
                f"angle_tolerance={round(self.angle_tolerance, 2)}, "
                f"bidirectional={self.bidirectional}, num_goals={self.num_goals}, "
                f"debug={self.debug})")

    def step(self, state):
        if not self.got_waypoint or (self.got_waypoint and self.reached_waypoint):
            return
        if sqrt((state['car_obj'].position[0] - self.waypoints[0][0])**2 + (state['car_obj'].position[1] - self.waypoints[0][1])**2) < 2:
            self.reached_waypoint = True

    def get_unified_state(self):
        # Format goals in the same way as EnvironmentGoal for compatibility with GoalStop
        goals_formatted = []
        for car_x, car_y, car_direction in self.selected_goals:
            goals_formatted.append([car_x, car_y, car_direction])

        return {
            'name': 'goal_module',
            'goals': self.waypoints if self.got_waypoint and not self.reached_waypoint else goals_formatted,
            'bidirectional': self.bidirectional,
            'goal_size': self.goal_size,
            'angle_tolerance': self.angle_tolerance
        }


class BarrierSpawner(GenericEnvironment):
    def __init__(self, barrier_distance=3.0, spawn_probability=0.0,
                 barrier_width=40, barrier_thickness=1, debug=False):
        super().__init__()
        self.barrier_distance = barrier_distance
        self.spawn_probability = spawn_probability
        self.barrier_width = barrier_width
        self.barrier_thickness = barrier_thickness
        self.debug = debug

        self.barriers = []
        self.barrier_obstacles = []

    def reset(self, mode, state):
        if mode != 'environment':
            return

        # Find goal module and lot orientation
        found_goals = False
        found_planner = False
        goals = []
        lot_orientation = None

        for env_module in state['environment']:
            if env_module.get('name') == 'goal_module':
                found_goals = True
                goals = env_module.get('goals', [])
            elif env_module.get('name') == 'lot_planner_module':
                found_planner = True
                lot_orientation = env_module.get('lot_orientation')

        if not found_goals:
            raise KeyError("BarrierSpawner: goal_module not found in state")
        if not found_planner:
            raise KeyError("BarrierSpawner: lot_planner_module not found in state")

        if not goals:
            self.barriers = []
            self.barrier_obstacles = []
            return

        # Clear previous barriers
        self.barriers = []
        self.barrier_obstacles = []

        # Find furthest goal from origin
        furthest_goal = None
        max_distance = 0

        for goal_x, goal_y, goal_direction in goals:
            distance = sqrt(goal_x ** 2 + goal_y ** 2)
            if distance > max_distance:
                max_distance = distance
                furthest_goal = (goal_x, goal_y, goal_direction)

        if furthest_goal is None:
            return

        # Spawn barrier with probability
        if random() < self.spawn_probability:
            goal_x, goal_y, goal_direction = furthest_goal

            # Offset along lot axis, away from origin
            if lot_orientation == 'vertical':
                # Vertical lots run along y-axis, so offset in y direction
                # Use sign of goal_y to determine offset direction
                offset_direction = 1 if goal_y >= 0 else -1
                barrier_x = goal_x
                barrier_y = goal_y + offset_direction * self.barrier_distance
                barrier_direction = 0  # Horizontal barrier for vertical lot
            else:  # horizontal
                # Horizontal lots run along x-axis, so offset in x direction
                # Use sign of goal_x to determine offset direction
                offset_direction = 1 if goal_x >= 0 else -1
                barrier_x = goal_x + offset_direction * self.barrier_distance
                barrier_y = goal_y
                barrier_direction = pi / 2  # Vertical barrier for horizontal lot

            # Store barrier info
            self.barriers.append((barrier_x, barrier_y, barrier_direction))

            # Create collision obstacle
            barrier_obstacle = RectObstacle(
                [barrier_x, barrier_y],
                [self.barrier_width, self.barrier_thickness],
                angle=180 / pi * barrier_direction
            )
            self.barrier_obstacles.append(barrier_obstacle)

    def render(self, screen, transform):
        # Render barrier surfaces
        for barrier_x, barrier_y, barrier_direction in self.barriers:
            self._render_barrier_surface(screen, transform, barrier_x, barrier_y, barrier_direction)

        if self.debug:
            self._render_debug_info(screen, transform)

    def _render_barrier_surface(self, screen, transform, x, y, direction):
        # Add 90 degrees to rendering direction to match collision object
        render_direction = direction + pi / 2
        cos_dir = cos(render_direction)
        sin_dir = sin(render_direction)
        half_width = self.barrier_width / 2
        half_thickness = self.barrier_thickness / 2

        corners = [
            [-half_thickness, -half_width],
            [half_thickness, -half_width],
            [half_thickness, half_width],
            [-half_thickness, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        # Draw grey filled rectangle for barrier
        pygame.draw.polygon(screen, (90, 90, 90), screen_corners)

        # Draw darker grey border
        pygame.draw.polygon(screen, (60, 60, 60), screen_corners, 2)

    def _render_debug_info(self, screen, transform):
        for barrier_x, barrier_y, barrier_direction in self.barriers:
            center_screen = transform @ np.array([barrier_x, barrier_y, 1])
            pygame.draw.circle(screen, (255, 0, 255),
                               (int(center_screen[0]), int(center_screen[1])), 5)

    def get_digest(self):
        return (f"BarrierSpawner(barrier_distance={self.barrier_distance}, "
                f"spawn_probability={self.spawn_probability}, "
                f"barrier_width={self.barrier_width}, "
                f"barrier_thickness={self.barrier_thickness}, "
                f"debug={self.debug})")

    def get_unified_state(self):
        return {
            'name': 'barrier_spawner_module',
            'obstacles': self.barrier_obstacles,
            'barriers': self.barriers,
            'barrier_count': len(self.barriers)
        }
    

class CurbRenderer(GenericEnvironment):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.curb_spawns = []
        self.curb_obstacles = []

    def reset(self, mode, state):
        if mode != 'environment':
            return

        found_planner = False
        for env_module in state['environment']:
            if env_module.get('name') == 'lot_planner_module':
                found_planner = True
                spawn_points = env_module.get('spawn_points', [])
                self.curb_spawns = [sp for sp in spawn_points if sp[0] == 'curb']
                break

        if not found_planner:
            raise KeyError("CurbRenderer: lot_planner_module not found in state")

        # Create collision obstacles for curbs
        self.curb_obstacles = []
        for spawn_type, x, y, direction, width, thickness in self.curb_spawns:
            # Adjust position to center the rectangle - move half width along direction
            cos_dir = cos(direction)
            sin_dir = sin(direction)
            center_x = x + (width / 2) * sin_dir
            center_y = y + (width / 2) * cos_dir

            curb_obstacle = RectObstacle(
                [center_x, center_y],
                [width, thickness],
                angle=180 / pi * (direction + pi / 2)  # Perpendicular to lot direction
            )
            self.curb_obstacles.append(curb_obstacle)

    def render(self, screen, transform):
        # Render curb surfaces
        for spawn_type, x, y, direction, width, thickness in self.curb_spawns:
            self._render_curb_surface(screen, transform, x, y, direction, width, thickness)

        if self.debug:
            self._render_debug_info(screen, transform)

    def _render_curb_surface(self, screen, transform, x, y, direction, width, thickness):
        # Adjust to center position
        cos_dir = cos(direction)
        sin_dir = sin(direction)
        center_x = x + (width / 2) * sin_dir
        center_y = y + (width / 2) * cos_dir
        half_width = width / 2
        half_thickness = thickness / 2

        corners = [
            [-half_thickness, -half_width],
            [half_thickness, -half_width],
            [half_thickness, half_width],
            [-half_thickness, half_width]
        ]

        world_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_dir - corner_y * sin_dir + center_x
            rotated_y = corner_x * sin_dir + corner_y * cos_dir + center_y
            world_corners.append([rotated_x, rotated_y, 1])

        screen_corners = []
        for corner in world_corners:
            screen_corner = transform @ np.array(corner)
            screen_corners.append((int(screen_corner[0]), int(screen_corner[1])))

        # Draw grey filled rectangle for curb
        pygame.draw.polygon(screen, (128, 128, 128), screen_corners)

        # Draw darker grey border
        pygame.draw.polygon(screen, (80, 80, 80), screen_corners, 2)

    def _render_debug_info(self, screen, transform):
        for spawn_type, x, y, direction, width, thickness in self.curb_spawns:
            center_screen = transform @ np.array([x, y, 1])
            pygame.draw.circle(screen, (255, 0, 0),
                               (int(center_screen[0]), int(center_screen[1])), 5)

    def get_digest(self):
        return f"CurbRenderer(debug={self.debug})"

    def get_unified_state(self):
        return {
            'name': 'curb_renderer_module',
            'obstacles': self.curb_obstacles,
            'curb_count': len(self.curb_spawns)
        }


def parking_env2(render=False, goal_size=1.0, episodes=200, spot_width=4, road_width=10, decimation=0.5, vision=False, vision_model=None, force_first_car=False, angle_tolerance=pi / 4,
                 generate_curb=False, curb_depth=1.0, barrier_distance=3.0, spawn_probability=0.0, env_passthrough=None):
    world_width = 90
    world_aspect = 3 / 4
    env_passthrough = {} if env_passthrough is None else env_passthrough
    env_passthrough['world_aspect'] = env_passthrough.get('world_aspect', world_aspect)
    env_passthrough['world_width'] = env_passthrough.get('world_width', world_width)

    environment_modules = [EnvironmentSelector(),
                           LotEnvironmentPlanner(main_road_width=road_width, margin=2, curb_enabled=generate_curb, curb_depth=curb_depth),
                           RoadRenderer(),
                           ParkedCarRow(parking_spot_width=spot_width, vacancy_rate=decimation, force_first_car=force_first_car),
                           GoalSelector(goal_size=goal_size, num_goals=1, angle_tolerance=angle_tolerance),
                           BarrierSpawner(barrier_distance=barrier_distance, spawn_probability=spawn_probability),
                           CurbRenderer(),
                           Borders(),
                           ]

    stop_conditions = [StepLimit(step_limit=episodes),
                       CollisionStop(),
                       GoalStop()
                       ]

    reward_functions = [GoalEndReward(),
                        TimePenalty(),
                        CollisionPenalty(reward=-1),
                        DistanceReward(),
                        ObstacleProximityReward(),
                        RewardDisplayModule()
                        ]

    if vision_model is not None:
        vision = True

    observation_module = [ClassicalObservation()]
    if vision_model is not None:
        observation_module.append(VisionRaycastObservation(vision_model, show_image=False))

    env = load_env(render=render,
                   stop_conditions=stop_conditions, environment_modules=environment_modules, reward_functions=reward_functions,
                   observation_modules=observation_module, generate_vision=vision,
                   max_ray_distance=10, rays=24, **(env_passthrough if env_passthrough is not None else {}))
    return env


class ParkingSchedule2(GenericTrainingSchedule):
    def __init__(self, render=False, start_from_env=0, vision=False, vision_model=None):
        super().__init__(start_from_env=start_from_env)
        self.environments = [parking_env2(render=render, goal_size=1.5, decimation=0.95, angle_tolerance=pi / 4, force_first_car=False, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=1.5, decimation=0.9, angle_tolerance=pi / 8, force_first_car=False, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=1.5, episodes=250, decimation=0.8, force_first_car=False, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=1.0, episodes=300, decimation=0.7, force_first_car=False, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=1.0, episodes=350, decimation=0.6, force_first_car=False, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=0.7, episodes=350, decimation=0.3, angle_tolerance=pi / 10, curb_depth=4, generate_curb=True, vision=vision, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=0.7, episodes=200, decimation=0.3, angle_tolerance=pi / 10, generate_curb=True, spawn_probability=0.6, vision=vision, vision_model=vision_model, env_passthrough={'world_aspect': 9/16}),
                             parking_env2(render=render, goal_size=0.7, episodes=200, decimation=0.3, angle_tolerance=pi / 10, curb_depth=0, generate_curb=True, spawn_probability=0.6, vision=vision, vision_model=vision_model, env_passthrough={'world_aspect': 9/16}),
                             ]

        base_params = {
            'num_envs': min(14, max(1, cpu_count())),
            'action_dim': 2,
            'batch_size': 512,
            'total_timesteps': 2_000_000,
            'save_freq': 20000,
            'eval_episodes': 10,
            'seed': 41,
            'exploration_noise': 0.15,
            'start_timesteps': 25000,
            'buffer_size': 1_000_000,
            'learning_rate': 3e-4,
            'net_size': [400, 300],
        }
        self.parameters = [base_params, {'start_timesteps': 25000, 'total_timesteps': 2_000_000}, {}, {}]


class ParkingScheduleDeployment(GenericTrainingSchedule):
    def __init__(self, render=False, decimation=0.3, vision=False, vision_model=None):
        super().__init__(start_from_env=0)
        self.environments = [parking_env2(render=render, goal_size=1, episodes=500, decimation=decimation, angle_tolerance=pi / 6, curb_depth=0, generate_curb=True,  force_first_car=True, spawn_probability=0.0, vision=vision,
                                          vision_model=vision_model, env_passthrough={'screen_width': 800, 'world_aspect': 9/16, 'world_width': 110})]

        base_params = {
            'num_envs': min(14, max(1, cpu_count())),
            'action_dim': 2,
            'batch_size': 512,
            'total_timesteps': 2_000_000,
            'save_freq': 20000,
            'eval_episodes': 10,
            'seed': 41,
            'exploration_noise': 0.15,
            'start_timesteps': 25000,
            'buffer_size': 1_000_000,
            'learning_rate': 3e-4,
            'net_size': [400, 300],
        }
        self.parameters = [base_params]
