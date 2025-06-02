from modules.generic_modules import GenericEnvironment
from Objects.obstacles import RectObstacle
from Objects.car import Car
from random import choice, randint, sample
import pygame


class Borders(GenericEnvironment):
    def __init__(self, wall_width=2):
        super().__init__()
        self.wall_width = wall_width
        self.world_width = None
        self.world_height = None
        self.collision_rects = []

    def reset(self, mode, state=None):
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        self.collision_rects = [RectObstacle([0, self.world_height/2 - self.wall_width/2], [self.world_width, self.wall_width]),
                            RectObstacle([0, -self.world_height/2 + self.wall_width/2], [self.world_width, self.wall_width]),
                            RectObstacle([self.world_width/2 - self.wall_width/2, 0], [self.wall_width, self.world_height]),
                            RectObstacle([-self.world_width/2 + self.wall_width/2, 0], [self.wall_width, self.world_height])]

    def render(self, screen, transform_matrix):
        for rect in self.collision_rects:
            rect.draw(screen, transform_matrix)

    def get_digest(self):
        return f"Borders(world_width={self.world_width}, world_height={self.world_height}, "\
               f"wall_width={self.wall_width})"

    def get_unified_state(self):
        #print(f"DEBUG: Borders returning {len(self.collision_rects)} wall obstacles")#
        return {'obstacles': self.collision_rects}


## ============== Parking Lot Environment =================
## =============== Credit: Nikhil & Jack ==================

class ParkingLotModule(GenericEnvironment):
    def __init__(self, wall_width=2, configuration=1, generate_goals=True, goal_radius = 1.4):
        super().__init__()
        self.configuration = configuration
        self.wall_width = wall_width
        self.world_width = None
        self.world_height = None
        self.collision_rects = []
        self.static_cars = []
        self.obstacles = []
        self.goals = []
        self.generate_goals = generate_goals
        self.goal_radius =goal_radius 

    def reset(self, mode, state=None):
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        self.collision_rects = []
        self.goals = []

        # Available colors for cars
        colors = [(240, 230, 0), (0, 255, 0), (0, 233, 255)]

        if self.configuration == 0:
            # Configuration 0: Cars in columns (vertical arrangement)
            # Adjusted coordinates to accommodate wider obstacles (6.5 width instead of 4.7)
            coordinate_pool = [
                [-24, 6], [-24, 0], [-24, -6], [-24, -12], [-24, 12],
                [24, 6], [24, 0], [24, -6], [24, -12], [24, 12],
                [-24, 15], [-24, 3], [-24, -3], [-24, -9], [-24, 9], [-24, -15],
                [24, 15], [24, 3], [24, -3], [24, -9], [24, 9], [24, -15],
                [-24, 18], [-24, -18], [24, 18], [24, -18]
            ]

            # Ensure more cars than obstacles: max obstacles is 1/3 of total positions
            min_obstacles = 3
            max_obstacles = min(8, len(coordinate_pool) // 3)  # Cap at 8 or 1/3 of positions
            num_obstacles = randint(min_obstacles, max_obstacles)

            obstacle_positions = sample(coordinate_pool, num_obstacles)
            car_positions = [pos for pos in coordinate_pool if pos not in obstacle_positions]

            self.obstacles = []
            for pos in obstacle_positions:
                self.obstacles.append(RectObstacle(pos, [4.7, 2.5]))  # Wider obstacles
                self.goals.append((*pos, 0))

            self.static_cars = []
            for pos in car_positions:
                color = choice(colors)
                self.static_cars.append(Car(origin=pos, start_direction=0, color=color))  # << Car is an *extremely* expensive class to use for just rendering.

                # Consider 'from Objects.car import render_car', and just using that instead. -delta
                #  for i, blocker in enumerate(self.collision_rects):
                #    width, length = blocker.size
                #    render_car(screen, transform, blocker.position, car_angle=180 - blocker.angle - 90, width=width, length=length, color=color)

        elif self.configuration == 1:
            # Configuration 1: Cars in rows (horizontal arrangement)
            # Adjusted coordinates to accommodate wider obstacles (6.5 width instead of 4.7)
            coordinate_pool = [
                [-24, 10], [-18, 10], [-12, 10], [-6, 10], [0, 10], [6, 10], [12, 10], [18, 10], [24, 10],
                [-24, -10], [-18, -10], [-12, -10], [-6, -10], [0, -10], [6, -10], [12, -10], [18, -10], [24, -10],
                [-21, 10], [-15, 10], [-9, 10], [-3, 10], [3, 10], [9, 10], [15, 10], [21, 10],
                [-21, -10], [-15, -10], [-9, -10], [-3, -10], [3, -10], [9, -10], [15, -10], [21, -10]
            ]
             # Ensure more cars than obstacles: max obstacles is 1/3 of total positions
            min_obstacles = 3
            max_obstacles = min(10, len(coordinate_pool) // 3)  # Cap at 10 or 1/3 of positions
            num_obstacles = randint(min_obstacles, max_obstacles)

            obstacle_positions = sample(coordinate_pool, num_obstacles)
            car_positions = [pos for pos in coordinate_pool if pos not in obstacle_positions]

            self.obstacles = []
            for pos in obstacle_positions:
                self.obstacles.append(RectObstacle(pos, [4.7, 2.5], angle=90))  # Wider obstacles. Also, this expects radians
                self.goals.append((*pos, 0))

            self.static_cars = []
            for pos in car_positions:
                color = choice(colors)
                self.static_cars.append(Car(origin=pos, start_direction=90, color=color)) # Would recommend merging this with the above code, so you dont have two copies of it

    def render(self, screen, transform_matrix):
        for rect in self.collision_rects:
            rect.draw(screen, transform_matrix)

        for car in self.static_cars:
            car.draw(screen, transform_matrix)

        for obstacle in self.obstacles:
            obstacle.draw(screen, transform_matrix)

    def get_digest(self):  # The digest is kinda like an instruction manual so anyone else can put together the same environment. Here, you'd need to put in the configuration parameter from up top
        return f"ParkingLotModule(world_width={self.world_width}, world_height={self.world_height}, "\
               f"wall_width={self.wall_width}, generate_goals = {self.generate_goals})"  # ClassName(parameter={parameter} ... )

    def get_unified_state(self):
        # Start with walls (always collidable)
        all_obstacles = self.collision_rects.copy()

        # ADD: Only static cars as collision objects (not the obstacles)
        i=0
        for car in self.static_cars:
            car_collision_box = RectObstacle(
                position=car.position,
                size=[car.length, car.width],
                angle=car.get_angle(),
                color=(255, 0, 0)
            )
            all_obstacles.append(car_collision_box)
            #print(f"DEBUG: Car {i} at {car.position} -> collision box at {car_collision_box.position}")
            i = i + 1

        #print(f"DEBUG: ParkingLotModule returning {len(all_obstacles)} obstacles")

            # DON'T add self.obstacles - these are parking spaces, not collision objects
        if not self.goals:
            self.goals = [(1000,1000,0)]
        return {
            'name' : 'goal_module',
            'obstacles': all_obstacles,      # Walls + Cars (collidable)
            'static_cars': self.static_cars,
            'static_obstacles': self.obstacles,  # Parking spaces (visual only)
            'goals' : self.goals if self.generate_goals else [],
            'goal_size' : self.goal_radius,
            'angle_tolerance' : 999,
            'bidirectional' : False
        }
    


class RandomSpawnModule(GenericEnvironment):
    def __init__(self, spawn_distance_range=(15, 25), diagonal_offset_range=(-8, 8)):
        super().__init__()
        self.spawn_distance_range = spawn_distance_range
        self.diagonal_offset_range = diagonal_offset_range
        self.world_width = None
        self.world_height = None
        self.spawn_position = None
        self.spawn_angle = None
        self.should_apply_spawn = False

    def reset(self, mode, state=None):
        import random
        
        # Get world size
        if state and 'world_size' in state:
            self.world_width = state['world_size'][0]
            self.world_height = state['world_size'][1]
        
        # Generate random spawn along vertical diagonal
        distance_factor = random.uniform(0.3, 0.9)
        
        # Calculate position along diagonal (bottom-left to top-right)
        diagonal_x = (self.world_width * distance_factor - self.world_width/2) 
        diagonal_y = (self.world_height * distance_factor - self.world_height/2)
        
        # Add random offset perpendicular to diagonal
        offset_distance = random.uniform(*self.diagonal_offset_range)
        perp_x = -diagonal_y / (self.world_height/2) * offset_distance
        perp_y = diagonal_x / (self.world_width/2) * offset_distance
        
        # Final spawn position and angle
        self.spawn_position = [diagonal_x + perp_x, diagonal_y + perp_y]
        self.spawn_angle = random.uniform(0, 2 * 3.14159)  # Random angle in radians
        self.should_apply_spawn = True
        
        #print(f"Random spawn: pos=({self.spawn_position[0]:.1f}, {self.spawn_position[1]:.1f}), angle={self.spawn_angle:.2f} rad")

    def apply_spawn_to_car(self, car):
        """Apply the spawn position and orientation to the car"""
        if self.should_apply_spawn and self.spawn_position and self.spawn_angle is not None:
            car.position = self.spawn_position.copy()
            car.direction_vector = car.angular_direction_to_vector(self.spawn_angle * 180 / 3.14159)  # Convert to degrees
            self.should_apply_spawn = False
            #print(f"Applied spawn: car now at {car.position} with direction {car.direction_vector}")

    def render(self, screen, transform_matrix):
        # Draw spawn point for debugging
        if self.spawn_position:
            import numpy as np
            spawn_screen = transform_matrix @ np.array([*self.spawn_position, 1])
            pygame.draw.circle(screen, (255, 255, 0), 
                             (int(spawn_screen[0]), int(spawn_screen[1])), 5)

    def get_digest(self):
        return f"RandomSpawnModule(distance_range={self.spawn_distance_range}, offset_range={self.diagonal_offset_range})"

    def get_unified_state(self):
        return {
            'car_position': self.spawn_position,
            'car_orientation': self.spawn_angle,
            'spawn_position': self.spawn_position,
            'spawn_angle': self.spawn_angle,
            'random_spawn_module': self  # Pass reference to module
        }