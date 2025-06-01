from modules.generic_modules import GenericEnvironment, GenericStop
from AI.YOLO.yolo_detect import YOLODetector
import pygame
import numpy as np
import math
from os.path import join, exists


class YOLOGoalDetector(GenericEnvironment):
    def __init__(self, model_name, search_path="models", confidence_threshold=0.5):
        super().__init__()
        self.model_name = model_name
        self.search_path = search_path
        self.confidence_threshold = confidence_threshold
        self.world_width = None
        self.world_height = None
        self.parking_goals = []
        self.detected_objects = []

        yolo_model_path = self._get_model_path()
        self.yolo_detector = YOLODetector(yolo_model_path, self.confidence_threshold)

        self.detection_frame_counter = 0
        self.detection_interval = 30
        self.goals_detected = False
        self.total_detections_run = 0

    def _get_model_path(self):
        if self.model_name is None:
            raise ValueError("model_name is required")

        model_path = join(self.search_path, "YOLO", self.model_name, "weights", "best.pt")

        if not exists(model_path):
            raise ValueError(f"YOLO model not found at {model_path}")

        return model_path

    def reset(self, mode, state):
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]

        self.parking_goals = []
        self.detected_objects = []
        self.detection_frame_counter = 0
        self.goals_detected = False
        self.total_detections_run = 0

    def render(self, screen, transform_matrix):
        if not self.yolo_detector.is_active():
            return

        self.detection_frame_counter += 1
        interval = 5 if not self.goals_detected else self.detection_interval

        if self.detection_frame_counter >= interval:
            self.detection_frame_counter = 0
            self.total_detections_run += 1
            self._run_yolo_detection_from_screen(screen)

    def _run_yolo_detection_from_screen(self, screen):
        try:
            if screen is None:
                return

            detections = self.yolo_detector.detect_from_pygame_screen(screen)
            self.detected_objects = detections
            self.parking_goals = []

            parking_spots_found = 0
            for detection in detections:
                if detection['class_name'] == "ParkingSpot":
                    parking_spots_found += 1
                    goal = self._create_parking_goal(detection, screen.get_size())
                    if goal:
                        self.parking_goals.append(goal)

            if self.parking_goals and not self.goals_detected:
                self.goals_detected = True

        except Exception as e:
            print(f"YOLO detection error: {e}")
            import traceback
            traceback.print_exc()

    def _create_parking_goal(self, detection, screen_size):
        screen_width, screen_height = screen_size
        screen_center_x = detection['center'][0]
        screen_center_y = detection['center'][1]

        world_x, world_y = self._screen_to_world_coords(
            screen_center_x, screen_center_y, screen_width, screen_height
        )

        angle_to_center = self._calculate_angle_to_center(world_x, world_y)
        goal_x = world_x
        goal_y = world_y

        goal = {
            'position': [goal_x, goal_y],
            'angle': angle_to_center,
            'size': [1.5, 1.5],
            'confidence': detection['confidence'],
            'bidirectional': True
        }

        return goal

    def _screen_to_world_coords(self, screen_x, screen_y, screen_width, screen_height):
        norm_x = (screen_x - screen_width / 2) / (screen_width / 2)
        norm_y = (screen_y - screen_height / 2) / (screen_height / 2)

        world_x = norm_x * (self.world_width / 2)
        world_y = norm_y * (self.world_height / 2)

        return world_x, world_y

    def _calculate_angle_to_center(self, x, y):
        if x == 0 and y == 0:
            return 0

        angle_rad = math.atan2(-y, -x)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def get_digest(self):
        return f"YOLOGoalDetector(model_name={self.model_name}, confidence_threshold={self.confidence_threshold})"

    def get_unified_state(self):
        formatted_goals = []
        for goal in self.parking_goals:
            formatted_goals.append((
                goal['position'][0],
                goal['position'][1],
                math.radians(goal['angle'])
            ))

        return {
            'name': 'YOLOGoals',
            'goals': formatted_goals,
            'parking_goals': self.parking_goals,
            'detected_objects': self.detected_objects
        }


class YOLOGoalStop(GenericStop):
    """
    YOLO Goal Stop that provides placeholder goals during initialization
    """

    def __init__(self, goal_radius=1.5):
        super().__init__()
        self.goal_radius = goal_radius
        self.goals_from_yolo = []
        self.placeholder_provided = False

    def reset(self, mode, state=None):
        """Reset and provide initial placeholder goals"""
        self.goals_from_yolo = []
        self.placeholder_provided = True  # Always provide placeholder initially

        # Try to get real YOLO goals if available
        if state and 'environment' in state:
            for module_state in state['environment']:
                if hasattr(module_state, 'get') and module_state.get('name') == 'YOLOGoals':
                    yolo_goals = module_state.get('goals', [])
                    if yolo_goals:
                        self.goals_from_yolo = yolo_goals
                        self.placeholder_provided = False
                    break

        if not self.goals_from_yolo:
            print("YOLOGoalStop: Providing placeholder goals, waiting for YOLO detection...")

    def check_stop(self, state):
        """Check if car has reached any goal"""
        # Update goals from current state
        self._update_goals_from_state(state)

        # If we only have placeholder goals, don't stop for them
        if self.placeholder_provided and not self.goals_from_yolo:
            return False, ""

        if not self.goals_from_yolo:
            return False, ""

        car_position = np.array(state['car']['position'])

        # Check distance to each detected goal
        for goal_x, goal_y, goal_angle in self.goals_from_yolo:
            goal_position = np.array([goal_x, goal_y])
            distance = np.linalg.norm(car_position - goal_position)

            if distance <= self.goal_radius:
                return True, f"Goal Hit"

        return False, ""

    def _update_goals_from_state(self, state):
        """Update goals from current state"""
        if not state or 'environment' not in state:
            return

        for module_state in state['environment']:
            if hasattr(module_state, 'get') and module_state.get('name') == 'YOLOGoals':
                new_goals = module_state.get('goals', [])
                if new_goals and (not self.goals_from_yolo or len(new_goals) != len(self.goals_from_yolo)):
                    self.goals_from_yolo = new_goals
                    self.placeholder_provided = False  # Real goals found
                break

    def render(self, screen, transform_matrix):
        """Render YOLO goals"""
        # Only render real goals, not placeholders
        if not self.placeholder_provided:
            for goal_x, goal_y, goal_angle in self.goals_from_yolo:
                goal_screen = transform_matrix @ np.array([goal_x, goal_y, 1])
                goal_screen_pos = (int(goal_screen[0]), int(goal_screen[1]))

                radius_world = self.goal_radius * 0.7
                radius_screen = max(4, int(radius_world * transform_matrix[0, 0]))

                # Draw goal circle (green with red border)
                pygame.draw.circle(screen, (0, 255, 0), goal_screen_pos, radius_screen, 2)
                pygame.draw.circle(screen, (255, 0, 0), goal_screen_pos, radius_screen, 1)

    def get_digest(self):
        return f"FixedYOLOGoalStop(goal_radius={self.goal_radius}, goals_count={len(self.goals_from_yolo)})"

    def get_unified_state(self):
        """Always provide goals - use placeholder if none detected yet"""
        if self.goals_from_yolo:
            return {
                'name': 'YOLOGoalStop',
                'goals': self.goals_from_yolo,
                'goal_radius': self.goal_radius
            }
        else:
            # Provide placeholder goals that won't trigger winning
            # Place them far from typical car spawn locations
            placeholder_goals = [
                (0, 0, 0),  # Center (safe default)
                (25, 20, 0),  # Corner positions (far from typical spawns)
                (-25, 20, 0),
                (25, -20, 0),
                (-25, -20, 0)
            ]
            return {
                'name': 'YOLOGoalStop',
                'goals': placeholder_goals,
                'goal_radius': self.goal_radius,
                'placeholder': True  # Flag indicating these are temporary
            }
