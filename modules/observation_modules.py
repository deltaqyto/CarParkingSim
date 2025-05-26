import numpy as np

from Utility.raycast import Ray, ray_cast

from modules.generic_modules import GenericObservation


class ClassicalObservation(GenericObservation):
    def __init__(self, rays=12, max_ray_distance=10):
        super().__init__()

    def get_observation(self, state, observation):
        observation = [
            *state['car']['observation'],
            *state['raycasts'],
            *state['closest_goal']['car_frame'],
        ]
        return observation

    def get_digest(self):
        return f'ClassicalObservation()'


import random
import string
import os
from PIL import Image
import numpy as np


class YOLOExtractor(GenericObservation):
    def __init__(self):
        super().__init__()

    def get_observation(self, state, observation):
        # Only trigger with 10% chance per frame
        if random.random() > 0.1:
            return observation

        if state['steps'] < 20:
            return observation  # Only get frames after stuff happens

        image = state['vision']  # pygame.surfarray.array3d(self.vision_surface)
        if image is None:
            raise ValueError("Vision is not enabled on simulation environment (use generate_vision=True)")

        for module in state['environment']:
            if module.get('name', '') != 'ParkingLot':
                continue
            cars = module['static_cars']
            spaces = module['static_obstacles']

            # Create output directories
            os.makedirs('yolo_output/images', exist_ok=True)
            os.makedirs('yolo_output/labels', exist_ok=True)

            # Generate random filename
            random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            image_path = f"yolo_output/images/{random_name}.jpg"
            annotation_path = f"yolo_output/labels/{random_name}.txt"

            # Save image (transpose and convert from pygame format)
            img_array = np.transpose(image, (1, 0, 2))
            img = Image.fromarray(img_array)
            img.save(image_path)

            # Prepare annotations
            annotations = []

            # Add main car (class 0)
            print(state['car'])
            car_aabb = state['car_obj'].get_aabb()
            car_annotation = self._world_to_yolo(car_aabb, 0)
            annotations.append(car_annotation)

            # Add static cars (class 2)
            for car in cars:
                car_aabb = car.get_aabb()
                car_annotation = self._world_to_yolo(car_aabb, 2)
                annotations.append(car_annotation)

            # Add parking spaces (class 1)
            for space in spaces:
                space_aabb = space.get_aabb()
                space_annotation = self._world_to_yolo(space_aabb, 1)
                annotations.append(space_annotation)

            # Save annotations
            with open(annotation_path, 'w') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')

            break  # Only process first ParkingLot module

        return observation

    def _world_to_yolo(self, world_aabb, class_id):
        """Convert world AABB to YOLO format"""
        min_x, min_y, max_x, max_y = world_aabb

        # Convert world coordinates to image coordinates
        # World origin (0,0) is at image center (400, 300)
        img_min_x = min_x + 400
        img_min_y = min_y + 300
        img_max_x = max_x + 400
        img_max_y = max_y + 300

        # Calculate center and dimensions in image coordinates
        center_x = (img_min_x + img_max_x) / 2
        center_y = (img_min_y + img_max_y) / 2
        width = img_max_x - img_min_x
        height = img_max_y - img_min_y

        # Normalize to [0,1] range
        norm_center_x = center_x / 800
        norm_center_y = center_y / 600
        norm_width = width / 800
        norm_height = height / 600

        return f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

    def get_digest(self):
        return f'YOLOExtractor()'
