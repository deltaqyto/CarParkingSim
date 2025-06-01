from modules.generic_modules import GenericEnvironment
from Objects.obstacles import RectObstacle
from Objects.car import Car
import os


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
from modules.generic_modules import GenericEnvironment
from Objects.obstacles import RectObstacle
from Objects.car import Car
from random import choice, randint, sample

# Change the configuration of the environment by changing the configuration parameter
class ParkingLotModule(GenericEnvironment):  # < Recommend renaming this to something else, and making a fresh module for it, instead of building on the same module
    def __init__(self, wall_width=2, configuration=1):  # < Configuration should be added to the digest below
        super().__init__()
        self.configuration = configuration
        self.wall_width = wall_width
        self.world_width = None
        self.world_height = None
        self.collision_rects = []
        self.static_cars = []
        self.obstacles = []

    def reset(self, mode, state=None):
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        self.collision_rects = []
        print("Adding static cars.. ")  # In your final code, strongly recommend you do not print anything except errors to the console, to avoid spamming it

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

            self.static_cars = []
            for pos in car_positions:
                color = choice(colors)
                self.static_cars.append(Car(origin=pos, start_direction=0, color=color))  # << Car is an *extremely* expensive class to use for just rendering. Consider 'from Objects.car import render_car', and just using that instead:
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
                self.obstacles.append(RectObstacle(pos, [4.7, 2.5], angle=90))  # Wider obstacles

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
               f"wall_width={self.wall_width})"  # ClassName(parameter={parameter} ... )

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
            
        return {
            'obstacles': all_obstacles,      # Walls + Cars (collidable)
            'static_cars': self.static_cars, 
            'static_obstacles': self.obstacles  # Parking spaces (visual only)
        }


from modules.generic_modules import GenericEnvironment
from yolo_detect import YOLODetector
import math
import pygame
import numpy as np
import cv2


class YOLOGoalDetector(GenericEnvironment):
    def __init__(self, yolo_model_path=None, confidence_threshold=0.5):
        super().__init__()
        self.world_width = None
        self.world_height = None
        self.parking_goals = []
        self.detected_objects = []
        
        # Get model path if not provided
        if yolo_model_path is None:
            yolo_model_path = self._get_model_path()
        
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector(yolo_model_path, confidence_threshold)
        
        # Track detection timing
        self.detection_frame_counter = 0
        self.detection_interval = 30  # Run detection every 30 frames
        self.goals_detected = False
        self.total_detections_run = 0

    def reset(self, mode, state=None):
        """Environment setup"""
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        
        # Reset parking goals and detected objects
        self.parking_goals = []
        self.detected_objects = []
        self.detection_frame_counter = 0
        self.goals_detected = False
        self.total_detections_run = 0
        #print("DEBUG: YOLOGoalDetector reset completed")

    def render(self, screen, transform_matrix):
        """
        Use the render method to capture screen and run YOLO detection
        This is called by SimulationEnvironment during rendering
        """
        if not self.yolo_detector.is_active():
            return
        
        self.detection_frame_counter += 1
        
        # Run detection periodically (every N frames)
        interval = 5 if not self.goals_detected else self.detection_interval
        
        if self.detection_frame_counter >= interval:
            self.detection_frame_counter = 0
            self.total_detections_run += 1
            #print(f"DEBUG: Running YOLO detection #{self.total_detections_run} via render method")
            self._run_yolo_detection_from_screen(screen)

    def _run_yolo_detection_from_screen(self, screen):
        """Run YOLO detection on the pygame screen"""
        try:
            if screen is None:
                #print("DEBUG: Screen is None")
                return
            
            #print(f"DEBUG: Screen size: {screen.get_size()}")
            
            # Run YOLO detection directly on the pygame screen
            detections = self.yolo_detector.detect_from_pygame_screen(screen)
            self.detected_objects = detections
            
            #print(f"DEBUG: YOLO found {len(detections)} total detections")
            
            # Clear previous parking goals
            self.parking_goals = []
            
            # Convert detections to parking goals
            parking_spots_found = 0
            for detection in detections:
                #print(f"DEBUG: Detection - Class: {detection['class_name']}, Confidence: {detection['confidence']:.2f}")
                if self._is_parking_related(detection['class_name']):
                    parking_spots_found += 1
                    goal = self._create_parking_goal(detection, screen.get_size())
                    if goal:
                        self.parking_goals.append(goal)
                        #print(f"DEBUG: Created goal at ({goal['position'][0]:.1f}, {goal['position'][1]:.1f})")
            
            #print(f"DEBUG: Found {parking_spots_found} parking spots, created {len(self.parking_goals)} goals")
            
            # Update goals detected status
            if self.parking_goals and not self.goals_detected:
                self.goals_detected = True
                print(f"YOLO: SUCCESS! Found {len(self.parking_goals)} parking goals from screen!")
            elif detections:
                detected_classes = [d['class_name'] for d in detections]
                #print(f"YOLO detected: {set(detected_classes)}, but no parking spots")
            else:
                print("YOLO: No detections found")
                
        except Exception as e:
            print(f"YOLO detection error: {e}")
            import traceback
            traceback.print_exc()

    def _is_parking_related(self, class_name):
        """Check if detected class is relevant for parking spots"""
        if class_name == 'ParkingSpot':
            #print(f"DEBUG: Found ParkingSpot class!")
            return True
        
        parking_classes = [
            'parkingspot', 'parking_spot', 'parking', 'parking_space', 
            'empty_space', 'slot'
        ]
        
        result = any(cls in class_name.lower() for cls in parking_classes)
        if result:
            #print(f"DEBUG: Found parking-related class: {class_name}")
            return result

    def _create_parking_goal(self, detection, screen_size):
        """Create a parking goal from YOLO detection - CENTER placement"""
        screen_width, screen_height = screen_size
        
        screen_center_x = detection['center'][0]
        screen_center_y = detection['center'][1]
        
        # Convert to world coordinates using screen dimensions
        world_x, world_y = self._screen_to_world_coords(
            screen_center_x, screen_center_y, screen_width, screen_height
        )
        
        #print(f"DEBUG: Parking spot at screen ({screen_center_x:.1f}, {screen_center_y:.1f}) -> world ({world_x:.2f}, {world_y:.2f})")
        
        # Calculate angle to face center of map
        angle_to_center = self._calculate_angle_to_center(world_x, world_y)
        
        # CENTER PLACEMENT - No offsets, place goal exactly at detected center
        goal_x = world_x
        goal_y = world_y
        
        #print(f"DEBUG: Goal placed at CENTER: ({goal_x:.2f}, {goal_y:.2f})")
        
        goal = {
            'position': [goal_x, goal_y],
            'angle': angle_to_center,
            'size': [1.5, 1.5],  # Larger size for center placement
            'confidence': detection['confidence'],
            'class_name': detection['class_name'],
            'bidirectional': True
        }
        
        return goal

    def _screen_to_world_coords(self, screen_x, screen_y, screen_width, screen_height):
        """Convert screen coordinates to world coordinates"""
        # Convert screen coordinates to normalized coordinates (-1 to 1)
        norm_x = (screen_x - screen_width / 2) / (screen_width / 2)
        norm_y = (screen_y - screen_height / 2) / (screen_height / 2)
        
        # Scale to world coordinates
        world_x = norm_x * (self.world_width / 2)
        world_y = norm_y * (self.world_height / 2)
        
        return world_x, world_y

    def _calculate_angle_to_center(self, x, y):
        """Calculate angle from position to center of map"""
        if x == 0 and y == 0:
            return 0
        
        angle_rad = math.atan2(-y, -x)
        angle_deg = math.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg

    def get_digest(self):
        return f"YOLOGoalDetector(goals_count={len(self.parking_goals)})"

    def get_unified_state(self):
        """Return formatted goals for the simulation"""
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

    def _get_model_path(self):
        """Get the path to the YOLO model file"""
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "runs", "detect", "train", "weights", "best.pt")
        
        print(f"Looking for YOLO model at: {model_path}")
        
        if os.path.exists(model_path):
            print(f"YOLO model found!")
            return model_path
        else:
            print(f"YOLO model not found, using default")
            return "yolov8n.pt"
