from modules.generic_modules import GenericStop
from AI.YOLO.yolo_detect import YOLODetector
import pygame
import numpy as np
import cv2
import math
from os.path import join, exists


class YOLOGoalStop(GenericStop):
    """YOLO Goal Stop with integrated detection using vision state"""

    def __init__(self, model_name, search_path="models", confidence_threshold=0.5, goal_radius=1.5):
        super().__init__()
        self.model_name = model_name
        self.search_path = search_path
        self.confidence_threshold = confidence_threshold
        self.goal_radius = goal_radius

        # YOLO detector setup
        yolo_model_path = self._get_model_path()
        self.yolo_detector = YOLODetector(yolo_model_path, self.confidence_threshold)

        # Detection state
        self.parking_goals = []
        self.detected_objects = []
        self.detection_frame_counter = 0
        self.detection_interval = 30
        self.goals_detected = False
        self.total_detections_run = 0
        self.world_width = None
        self.world_height = None

    def _get_model_path(self):
        if self.model_name is None:
            raise ValueError("model_name is required")

        model_path = join(self.search_path, "YOLO", self.model_name, "weights", "best.pt")

        if not exists(model_path):
            raise ValueError(f"YOLO model not found at {model_path}")

        return model_path

    def reset(self, mode, state=None):
        """Reset detection state"""
        if state and 'world_size' in state:
            self.world_width = state['world_size'][0]
            self.world_height = state['world_size'][1]

        self.parking_goals = []
        self.detected_objects = []
        self.detection_frame_counter = 0
        self.goals_detected = False
        self.total_detections_run = 0

    def check_stop(self, state):
        """Check if car has reached any goal and run YOLO detection"""
        # Run YOLO detection using vision state
        self._run_yolo_detection_from_vision(state)

        # Check if car reached any detected goal
        if not self.parking_goals:
            return False, ""

        car_position = np.array(state['car']['position'])

        # Check distance to each detected goal
        for goal in self.parking_goals:
            goal_position = np.array(goal['position'])
            distance = np.linalg.norm(car_position - goal_position)

            if distance <= self.goal_radius:
                return True, f"Goal Hit (confidence: {goal['confidence']:.2f})"

        return False, ""

    def _run_yolo_detection_from_vision(self, state):
        """Run YOLO detection using raw vision data from state"""
        self.detection_frame_counter += 1
        interval = 5 if not self.goals_detected else self.detection_interval

        if self.detection_frame_counter >= interval:
            self.detection_frame_counter = 0
            self.total_detections_run += 1

            # Get raw vision image (from pygame.surfarray.array3d - shape is (width, height, channels))
            vision_image = state['vision']

            # Transpose to standard image format (height, width, channels)
            vision_image = vision_image.transpose([1, 0, 2])

            # Convert RGB to BGR for YOLO detector
            if len(vision_image.shape) == 3 and vision_image.shape[2] == 3:
                vision_image_bgr = cv2.cvtColor(vision_image, cv2.COLOR_RGB2BGR)
            else:
                vision_image_bgr = vision_image

            # Run YOLO detection on BGR image
            detections = self.yolo_detector.detect_from_array(vision_image_bgr)
            self.detected_objects = detections
            self.parking_goals = []

            parking_spots_found = 0
            for detection in detections:
                if detection['class_name'] == "ParkingSpot":
                    parking_spots_found += 1
                    goal = self._create_parking_goal(detection, vision_image.shape)
                    if goal:
                        self.parking_goals.append(goal)

            if self.parking_goals and not self.goals_detected:
                self.goals_detected = True

    def _create_parking_goal(self, detection, image_shape):
        """Create parking goal from detection using image coordinates"""
        if len(image_shape) == 3:
            image_height, image_width = image_shape[:2]
        else:
            image_height, image_width = image_shape

        image_center_x = detection['center'][0]
        image_center_y = detection['center'][1]

        # Convert image coordinates to world coordinates
        world_x, world_y = self._image_to_world_coords(
            image_center_x, image_center_y, image_width, image_height
        )

        goal = {
            'position': [world_x, world_y],
            'angle': 0,
            'size': [1.5, 1.5],
            'confidence': detection['confidence'],
            'bidirectional': True
        }

        return goal

    def _image_to_world_coords(self, image_x, image_y, image_width, image_height):
        """Convert image coordinates to world coordinates"""
        if self.world_width is None or self.world_height is None:
            # Fallback if world size not available
            return 0, 0

        # Normalize image coordinates to [-1, 1]
        norm_x = (image_x - image_width / 2) / (image_width / 2)
        norm_y = (image_y - image_height / 2) / (image_height / 2)

        # Convert to world coordinates
        world_x = norm_x * (self.world_width / 2)
        world_y = norm_y * (self.world_height / 2)

        return world_x, world_y

    def render(self, screen, transform_matrix):
        """Render detected parking goals"""
        for goal in self.parking_goals:
            goal_x, goal_y = goal['position']
            goal_screen = transform_matrix @ np.array([goal_x, goal_y, 1])
            goal_screen_pos = (int(goal_screen[0]), int(goal_screen[1]))

            radius_world = self.goal_radius * 0.7
            radius_screen = max(4, int(radius_world * transform_matrix[0, 0]))

            # Draw goal circle (green with red border)
            pygame.draw.circle(screen, (0, 255, 0), goal_screen_pos, radius_screen, 2)
            pygame.draw.circle(screen, (255, 0, 0), goal_screen_pos, radius_screen, 1)

            # Draw confidence text
            font = pygame.font.Font(None, 24)
            confidence_text = font.render(f"{goal['confidence']:.2f}", True, (255, 255, 255))
            text_pos = (goal_screen_pos[0] - 15, goal_screen_pos[1] - 30)
            screen.blit(confidence_text, text_pos)

    def get_digest(self):
        return f"YOLOGoalStop(model_name={self.model_name}, confidence_threshold={self.confidence_threshold}, goal_radius={self.goal_radius})"

    def get_unified_state(self):
        """Return current goals and detection info"""
        # Ensure we always have at least one goal (fallback)
        if not self.parking_goals:
            self.parking_goals = [{
                'position': [2000, 2000],
                'angle': 0,
                'size': [1.5, 1.5],
                'confidence': 0.0,
                'bidirectional': True
            }]

        formatted_goals = []
        for goal in self.parking_goals:
            formatted_goals.append((
                goal['position'][0],
                goal['position'][1],
                math.radians(goal['angle'])
            ))

        return {
            'name': 'YOLOGoalStop',
            'goals': formatted_goals,
            'goal_radius': self.goal_radius,
            'parking_goals': self.parking_goals,
            'detected_objects': self.detected_objects,
            'total_detections_run': self.total_detections_run
        }
