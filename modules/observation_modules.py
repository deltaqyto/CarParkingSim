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
