from math import sin, cos, radians, sqrt, pi
from Utility.console_logger import ConsoleLogger
from Objects.obstacles import RectObstacle
import pygame
import numpy as np


def smoothstep(start, stop, steps, inclusive=False):
    inclusive_val = 1 if inclusive else 0
    step = (stop-start)/(steps - inclusive_val)
    return [start + step * n for n in range(steps)]


class Car:
    def __init__(self, origin=None, start_direction=0, width=2, length=4.7,                                                                      # was 10
                 throttle_limit=1, steering_limit=1, speed_limit = 5.5, throttle_effective_acceleration = 3, drag_coefficient=1, min_turn_radius=7, steering_speed=0.5,
                 discrete_input=True, discrete_throttle_steps=3, discrete_steering_steps=3, console_logger=None, color=(255, 0, 0)):
        self.origin = origin[:] if origin is not None else [0, 0]
        self.start_direction = start_direction

        self.speed = 0  # m/s, affected by throttle input
        self.current_steer = 0.0  # normalised [-1, 1]. Will slowly track the desired steer input
        self.position = origin if origin is not None else [0, 0]
        self.direction_vector = self.angular_direction_to_vector(start_direction)

        self.console_logger = ConsoleLogger(silent=True) if console_logger is None else console_logger

        self.throttle_limit = throttle_limit
        self.steering_limit = steering_limit

        self.discrete_input = discrete_input
        self.discrete_throttle = discrete_throttle_steps
        self.discrete_steer = discrete_steering_steps
        self.input_states = [[(th, st) for th in smoothstep(-self.throttle_limit, self.throttle_limit, discrete_throttle_steps, inclusive=True)]
                                       for st in smoothstep(-self.steering_limit, self.steering_limit, discrete_steering_steps, inclusive=True)]
        self.input_states = [a for b in self.input_states for a in b]

        if self.discrete_input:
            self.console_logger.debug(self, f"Car has {len(self.input_states)} input states")

        self.min_turn_radius = min_turn_radius  # Meters, used to determine max steering angle
        self.width = width
        self.length = length
        self.speed_limit = speed_limit
        self.throttle_effective_acceleration = throttle_effective_acceleration
        self.drag_coefficient = drag_coefficient
        self.steering_speed = steering_speed
        self.color = color

    @staticmethod
    def angular_direction_to_vector(angle):
        # 0 -> [1, 0], 90 -> [0, 1] ...
        return [cos(radians(angle)), sin(radians(angle))]

    @staticmethod
    def rotate(vector, angle):
        # rotates vector clockwise by degrees
        rad = radians(angle)
        cos_a = cos(rad)
        sin_a = sin(rad)

        x, y = vector
        new_x = x * cos_a + y * sin_a
        new_y = -x * sin_a + y * cos_a

        # Normalise
        length = sqrt(new_x ** 2 + new_y ** 2)
        if length > 0:
            new_x /= length
            new_y /= length

        return [new_x, new_y]

    def reset(self):
        self.position = self.origin
        self.speed = 0.0
        self.direction_vector = self.angular_direction_to_vector(self.start_direction)

    def stop(self):
        self.speed = 0.0

    def get_collision_rect(self):
        # Car's collision rectangle as (center_point(m), size(m), angle(deg)).
        angle = -np.arctan2(self.direction_vector[1], self.direction_vector[0]) * 180 / np.pi
        return self.position, (self.length, self.width), -angle

    def get_aabb(self):
        return RectObstacle(*self.get_collision_rect()).get_aabb()

    def get_corners(self):
        return RectObstacle(*self.get_collision_rect()).get_corners()

    def step(self, step_input, timestep=0.1):
        if self.discrete_input:
            if type(step_input) is not int:
                raise ValueError(f"Discrete step input is not of type int. Got type {type(step_input)}")
            if step_input >= len(self.input_states):
                raise ValueError(f"Discrete step input is too big ({len(self.input_states)}). Got {step_input}")
            if step_input < 0:
                raise ValueError(f"Discrete step input is too small. Got {step_input}")
            throttle, steer = self.input_states[step_input]
        else:
            if type(step_input) is np.ndarray:
                step_input = step_input.tolist()
            if type(step_input) not in [list, tuple]:
                raise ValueError(f"Continuous step input needs to be list/tuple of shape (throttle, steer). Got {step_input}")
            if len(step_input) < 2:
                raise ValueError(f"Not enough values in continuous step input. Require (throttle, steer), got {step_input}")
            throttle, steer = step_input[:2]

        # Clamp to limits
        throttle = max(-self.throttle_limit, min(self.throttle_limit, throttle))
        steer = max(-self.steering_limit, min(self.steering_limit, steer))
        self.console_logger.debug(self, f"Applying {throttle}, {steer} to car")
        self.evaluate_car_physics(steer, throttle, timestep)

    def evaluate_car_physics(self, steer, throttle, timestep):
        distance_travelled = self.speed * timestep

        self.speed += throttle * self.throttle_effective_acceleration * timestep
        self.speed -= self.drag_coefficient * self.speed * timestep
        self.speed = max(-self.speed_limit, min(self.speed_limit, self.speed))

        # Gradually turn toward target (simulates time to turn the wheel)
        if self.current_steer < steer:
            self.current_steer = min(self.current_steer + self.steering_speed * timestep, steer)
        elif self.current_steer > steer:
            self.current_steer = max(self.current_steer - self.steering_speed * timestep, steer)

        # Calculate max turn rate based on physics (speed/turn radius)
        max_physical_turn_rate = (abs(self.speed) / self.min_turn_radius) * (180 / pi) if abs(self.speed) > 0 else 0

        # Calculate actual turn rate (degrees per second)
        steering_direction = -1 if self.speed > 0 else 1
        turn_rate = self.current_steer * max_physical_turn_rate * steering_direction

        degrees_turned = turn_rate * timestep

        self.position = [p + distance_travelled * d for p, d in zip(self.position, self.direction_vector)]
        self.direction_vector = self.rotate(self.direction_vector, degrees_turned)

    def get_observation(self):
        return [*self.direction_vector, self.speed / self.speed_limit, self.current_steer]

    def get_unified_state(self):
        ob = self.get_observation()
        # assert len(ob) == len(self.get_observation_help())
        return {'observation': ob, 'num_observation': len(ob), 'position': self.position,
                'speed': self.speed, 'wheel_angle': self.current_steer,
                'angle': self.get_angle(),
                'direction_vector': self.direction_vector}

    def get_angle(self):
        return np.arctan2(self.direction_vector[1], self.direction_vector[0])

    @staticmethod
    def get_observation_help():
        # 4 values
        return ["direction ([x, y], normalised -1, 1)", "direction ([x, y], normalised -1, 1)", "speed (normalised to speed limit)", "steering angle (normalised -1, 1 of min turn radius)"]

    def get_action_size(self):
        if self.discrete_input:
            return len(self.input_states)
        return 2

    def draw(self, surface, transform_matrix):
        wheel_width_offset = 1.4  # 1.0 = wheels at car edge, >1.0 = wheels outside car, <1.0 = wheels inside car
        wheel_length_offset = 0.1  # position along car length (0.0 = front, 1.0 = back)

        # Calculate car dimensions and orientation
        car_angle = -np.arctan2(self.direction_vector[1], self.direction_vector[0]) * 180 / pi

        # Apply transform to car position
        pos_vec = np.array([self.position[0], self.position[1], 1])
        screen_pos = transform_matrix @ pos_vec
        screen_x, screen_y = int(screen_pos[0]), int(screen_pos[1])

        # Determine car rectangle dimensions after transform
        scale_x = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
        scale_y = np.sqrt(transform_matrix[1, 0] ** 2 + transform_matrix[1, 1] ** 2)
        car_width_px = int(self.width * scale_x)
        car_length_px = int(self.length * scale_y)

        # Calculate wheel dimensions
        wheel_width = max(3, int(car_width_px * 0.25))
        wheel_length = max(5, int(car_length_px * 0.2))

        # Calculate extended surface size to accommodate wheels that might extend beyond car body
        extended_width = car_width_px + wheel_width * 2 * max(0.0, wheel_width_offset - 1.0)

        # Create car body rectangle with extended size for wheels
        car_rect = pygame.Surface((car_length_px, extended_width), pygame.SRCALPHA)

        # Calculate offset to center car in the extended surface
        width_offset = int(wheel_width * max(0.0, wheel_width_offset - 1.0))

        # Draw car body at offset position
        pygame.draw.rect(car_rect, self.color,
                         (0, width_offset, car_length_px, car_width_px), 0, 3)

        # Add direction triangle (adjusted for offset)
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

        # Rear wheels (fixed direction)
        pygame.draw.rect(car_rect, wheel_color,
                         (rear_wheel_x, left_wheel_y, wheel_length, wheel_width))
        pygame.draw.rect(car_rect, wheel_color,
                         (rear_wheel_x, right_wheel_y, wheel_length, wheel_width))

        # Front wheels (turn based on steering)
        # Create wheel surface to rotate
        wheel_surf = pygame.Surface((wheel_length, wheel_width), pygame.SRCALPHA)
        pygame.draw.rect(wheel_surf, wheel_color, (0, 0, wheel_length, wheel_width))

        # Rotate the wheel surface based on steering angle (negative to match the visual direction)
        wheel_angle = -self.current_steer * 30  # Scale steering to visible angle
        rotated_wheel = pygame.transform.rotate(wheel_surf, wheel_angle)

        # Position front wheels
        wheel_rect = rotated_wheel.get_rect(center=(front_wheel_x + wheel_length // 2, left_wheel_y + wheel_width // 2))
        car_rect.blit(rotated_wheel, wheel_rect)

        wheel_rect = rotated_wheel.get_rect(center=(front_wheel_x + wheel_length // 2, right_wheel_y + wheel_width // 2))
        car_rect.blit(rotated_wheel, wheel_rect)

        # Rotate the entire car
        rotated_car = pygame.transform.rotate(car_rect, car_angle)

        # Get the new rectangle and position it at the car's position
        car_pos_rect = rotated_car.get_rect(center=(screen_x, screen_y))

        # Draw the car to the surface
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
        out =  (f"Car(position={self.position}, rotation={self.direction_vector}, color={self.color}, "
                f"width={self.width}, length={self.length}, speed_limit={self.speed_limit}, "
                f"steering_limit={self.steering_limit}, throttle_limit={self.throttle_limit}, "
                f"drag_coefficient={self.drag_coefficient}, turn_radius={self.min_turn_radius}, "
                f"acceleration={self.throttle_effective_acceleration}, steering_speed={self.steering_speed},"
                f"discrete={self.discrete_input}")
        out += f', discrete_steps={[self.discrete_throttle, self.discrete_steer]}' if self.discrete_input else ''

        out += ')'
        return out




if __name__ == "__main__":
    logger = ConsoleLogger('debug')

    # Initialize pygame and create a window
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Car Test")
    clock = pygame.time.Clock()

    # Create a car
    car = Car(origin=[0, 0], start_direction=0, color=(255, 50, 50),
              console_logger=logger, discrete_input=False)

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

        # Get keyboard input
        keys = pygame.key.get_pressed()
        throttle = 0
        steer = 0

        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0
        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = 1.0

        # Step the car simulation
        car.step([throttle, steer], 0.1)

        # Clear screen
        screen.fill((200, 200, 200))

        # Draw the car
        car.draw(screen, transform)

        # Draw some reference point at origin
        origin = transform @ np.array([0, 0, 1])
        pygame.draw.circle(screen, (0, 0, 0), (int(origin[0]), int(origin[1])), 5)

        # Update display
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
