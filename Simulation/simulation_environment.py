import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

from Utility.console_logger import ConsoleLogger
from Utility.raycast import Ray, ray_cast
from Utility.collision_system import CollisionSystem
from Objects.car import Car

from modules.environment_modules import Borders
from modules.reward_functions import GoalEndReward, TimePenalty, CollisionPenalty, DistanceReward
from modules.stop_conditions import omnidirectional_goal, StepLimit, CollisionStop


class SimulationEnvironment:
    def __init__(self, render=False, console_logger=None, discrete=False, screen_width=800,
                 reward_functions=None, environment_modules=None, world_width=60, world_aspect=3/4,
                 stop_conditions=None, car_params=None, delta_time=0.4, substeps=2,
                 rays=12, max_ray_distance=10):
        self.console_logger = ConsoleLogger('warning') if console_logger is None else console_logger
        self.world_size = [world_width, 0]
        self.world_aspect = world_aspect

        self.world_size[1] = self.world_size[0] * self.world_aspect
        self.delta_time = delta_time
        self.substeps = substeps
        self.rays = rays
        self.raycasts = [1 for _ in range(self.rays)]
        self.max_ray_distance = max_ray_distance

        self.observation = None
        self.observation_size = 19

        if car_params is None:
            car_params = {}
        self.discrete = discrete
        self.car = Car(discrete_input=discrete, **car_params)
        self.action_size = self.car.get_action_size()

        self.stop_conditions = stop_conditions
        self.environment_modules = environment_modules
        self.reward_functions = reward_functions
        if self.stop_conditions is None:
            self.stop_conditions = [StepLimit(step_limit=200), omnidirectional_goal()]
        if self.environment_modules is None:
            self.environment_modules = [Borders()]
        if self.reward_functions is None:
            self.reward_functions = [GoalEndReward()]

        self.collision_system = None
        self.collision_list = []

        self.running = False
        self.steps = 0
        self.state = None

        self.reset_environment()

        self.screen_width = screen_width
        self.screen_height = self.screen_width * self.world_aspect

        self.scale = min(self.screen_width / self.world_size[0],
                         self.screen_height / self.world_size[1])
        self.transform = np.array([
                [self.scale, 0, self.screen_width / 2],  # x scale, y shear, x translate
                [0, self.scale, self.screen_height / 2],  # x shear, y scale, y translate
                [0, 0, 1]  # perspective
            ])

        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Car Simulation Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

    def reset_environment(self):
        self.car.reset()
        self.steps = 0
        self.collision_system = CollisionSystem()
        self.state = self.get_unified_state()

        for module in self.environment_modules:
            module.reset('environment', state=self.state)

        for module in self.stop_conditions:
            module.reset('stop', state=self.state)

        for module in self.reward_functions:
            module.reset('reward', state=self.state)

        self.running = True

    def get_unified_state(self):
        environment_module_state = [module.get_unified_state() for module in self.environment_modules]
        obstacles = []
        for module in environment_module_state:
            obstacle = module.get('obstacles', [])
            obstacles += obstacle
        state = {
            'steps': self.steps,
            'car': self.car.get_unified_state(),
            'collision_module': self.collision_system,
            'collisions': self.collision_list,
            'environment': environment_module_state,
            'stop_conditions': [module.get_unified_state() for module in self.stop_conditions],
            'reward_functions': [module.get_unified_state() for module in self.reward_functions],
            'running': self.running,
            'world_size': self.world_size,
            'delta_time': self.delta_time,
            'substeps': self.substeps,
            'obstacles': obstacles,
            'raycasts': self.raycasts,
        }

        return state

    def get_observation(self):
        # Cast rays and get distances
        ray_distances = []
        car_heading = np.arctan2(self.car.direction_vector[1], self.car.direction_vector[0])

        for i in range(self.rays):
            angle = i * (2 * np.pi / self.rays) + car_heading
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            ray = Ray(self.car.position, ray_dir)
            distance, _, _ = ray_cast(ray, self.state['obstacles'])
            norm_distance = min(distance / self.max_ray_distance, 1.0)
            ray_distances.append(norm_distance)

        goals = []
        for module in self.state['stop_conditions']:
            goals += module.get('goals', [])
        if not goals:
            raise ValueError(f"No goals found in stop condition unified states")

        min_goal_distance = 1e6
        closest_goal = None
        car_angle = self.car.get_angle()

        # Create rotation matrix for transforming to car's frame
        cos_angle = np.cos(-car_angle)  # Negative angle to rotate world to car frame
        sin_angle = np.sin(-car_angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        for x, y, goal_angle in goals:
            # Calculate distance in world frame
            relative_goal_x = x - self.car.position[0]
            relative_goal_y = y - self.car.position[1]
            goal_distance = np.sqrt(relative_goal_x ** 2 + relative_goal_y ** 2)

            if goal_distance >= min_goal_distance:
                continue

            min_goal_distance = goal_distance

            # Transform goal position to car's frame
            relative_goal_world = np.array([relative_goal_x, relative_goal_y])
            relative_goal_car = rotation_matrix @ relative_goal_world

            # Also transform the goal angle to car's frame
            relative_goal_angle = (goal_angle - car_angle) % (2 * np.pi)
            # Normalize to [-π, π] range
            if relative_goal_angle > np.pi:
                relative_goal_angle -= 2 * np.pi

            # Normalize relative goal position
            max_distance = np.sqrt(self.world_size[0] ** 2 + self.world_size[1] ** 2)
            closest_goal = [
                relative_goal_car[0] / max_distance,
                relative_goal_car[1] / max_distance,
                relative_goal_angle
            ]

        self.raycasts = ray_distances
        self.state['raycasts'] = self.raycasts
        self.state['closest_goal'] = {'car_frame': closest_goal, 'distance': min_goal_distance}
        observation = [
            *self.state['car']['observation'],
            *ray_distances,
            *closest_goal,
        ]

        self.console_logger.debug(self, f"Sending observation: {observation}")
        return observation

    def step(self, action):
        prev_position = self.car.position.copy()
        prev_direction = self.car.direction_vector.copy()
        prev_speed = self.car.speed
        prev_steer = self.car.current_steer

        for _ in range(self.substeps):
            self.car.step(action, self.delta_time/self.substeps)

        self.collision_list = self.collision_system.collide_against(self.car)
        if self.collision_list:
            self.car.position = prev_position
            self.car.direction_vector = prev_direction
            self.car.speed = 0
            self.car.steer = prev_steer

        self.state = self.get_unified_state()

        stop_condition_triggered = False
        is_stopping = False
        reasons = []
        for module in self.stop_conditions:
            stop_condition_triggered, reason = module.check_stop(self.state)
            if stop_condition_triggered:
                is_stopping = True
                reasons.append(reason)
        stop_condition_triggered = is_stopping
        self.state['stop_reasons'] = reasons

        if self.render:
            self.render_frame()
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.complete_simulation(reason='User Quit')

        self.steps += 1
        if stop_condition_triggered:
            self.complete_simulation(reason=reasons)

        self.observation = self.get_observation()

        rewards = 0
        reward_types = {}
        for module in self.reward_functions:
            name, reward = module.get_reward(self.state)
            rewards += reward
            reward_types[name] = reward
        self.state['reward_types'] = reward_types

        # Is stopped?, observation, state
        return not self.running, self.observation, rewards, self.state

    def complete_simulation(self, reason):
        self.console_logger.info(self, f"Simulation terminated: {reason}")
        self.state['stop_reasons'].append(reason) if reason is not self.state['stop_reasons'] else None
        self.running = False

    def __del__(self):
        if self.render:
            pygame.display.quit()
            pygame.quit()

    def render_frame(self):
        if not self.render:
            return
        if not self.running:
            return

        # Clear screen
        self.screen.fill((200, 200, 200))

        for module in self.environment_modules:
            module.render(self.screen, self.transform)

        # Draw rays from the car
        car_heading = np.arctan2(self.car.direction_vector[1], self.car.direction_vector[0])
        ray_color = (0, 255, 0)  # Green color for rays

        for i in range(len(self.state['raycasts'])):
            # Calculate ray angle
            angle = i * (2 * np.pi / len(self.state['raycasts'])) + car_heading

            # Convert direction from car's direction vector
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            # Create ray starting from car position
            ray = Ray(self.car.position, ray_dir)

            # Cast ray and get collision info
            distance, hit_point, hit_object = ray_cast(ray, self.state['obstacles'])

            # Draw ray as line from car to hit point or max distance
            start_point = self.transform @ np.append(self.car.position, 1)
            end_point = self.transform @ np.append(hit_point[:2], 1)

            # Use different color if ray hit something
            line_color = (255, 0, 0) if hit_object is not None else ray_color

            pygame.draw.line(
                self.screen,
                line_color,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                2
            )

        for module in self.stop_conditions:
            module.render(self.screen, self.transform)

        # self.collision_system.draw_debug(self.screen, self.transform)

        self.car.draw(self.screen, self.transform)

        # Update display
        pygame.display.flip()

    def get_digest(self):
        reward_module_strings = []
        for module in self.reward_functions:
            reward_module_strings.append(module.get_digest())
        environment_module_strings = []
        for module in self.environment_modules:
            environment_module_strings.append(module.get_digest())
        stop_condition_strings = []
        for module in self.stop_conditions:
            stop_condition_strings.append(module.get_digest())

        out = (f"SimulationEnvironment(render={self.render}, discrete={self.discrete}, "
               f"screen_size={[self.screen_width, self.screen_height]}, "
               f"world_size={self.world_size}, world_aspect={self.world_aspect}, delta_time={self.delta_time}, substeps={self.substeps}, "
               f"rays={self.rays}, max_ray_distance={self.max_ray_distance}"
               f"): ["
               f"\n  reward_modules: [\n    " + '\n    '.join(reward_module_strings) +
               f"\n  ]\n  environment_modules: [\n    " + '\n    '.join(environment_module_strings) +
               f"\n  ]\n  stop conditions: [\n    " + '\n    '.join(stop_condition_strings) +
               f"\n  ]\n  car: " + self.car.get_digest() +
               f"\n]")

        return out
