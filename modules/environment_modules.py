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
        
        # Track screen for detection
        self.current_screen = None
        self.detection_frame_counter = 0
        self.detection_interval = 30  # Run detection every 30 frames to avoid performance issues

    def reset(self, mode, state=None):
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        
        # Reset parking goals and detected objects
        self.parking_goals = []
        self.detected_objects = []
        self.detection_frame_counter = 0
        self.goals_detected = False  # Reset detection status

    def update_detection(self, screen):
        """Update YOLO detection periodically"""
        self.current_screen = screen
        self.detection_frame_counter += 1
        
        # Run detection more frequently until goals are found, then less frequently
        interval = 1 if not self.goals_detected else 10  # Every frame until found, then every 10 frames
        
        if self.detection_frame_counter >= interval:
            self.detection_frame_counter = 0
            self._run_yolo_detection()

    def _run_yolo_detection(self):
        """Run YOLO detection and update parking goals"""
        if self.current_screen is None or not self.yolo_detector.is_active():
            return
        
        # Get detections from YOLO
        detections = self.yolo_detector.detect_from_pygame_screen(self.current_screen)
        self.detected_objects = detections
        
        # Clear previous parking goals
        self.parking_goals = []
        
        # Convert detections to parking goals
        for detection in detections:
            # Filter for relevant classes
            if self._is_parking_related(detection['class_name']):
                goal = self._create_parking_goal(detection)
                if goal:
                    self.parking_goals.append(goal)
        
        # Update goals detected status
        if self.parking_goals and not self.goals_detected:
            self.goals_detected = True
            print(f"YOLO: Found {len(self.parking_goals)} parking goals!")
        elif detections:
            detected_classes = [d['class_name'] for d in detections]
            print(f"YOLO detected: {set(detected_classes)}, but no parking spots")

    def _is_parking_related(self, class_name):
        """Check if detected class is relevant for parking spots"""
        parking_classes = [
            'parkingspot', 'parking_spot', 'parking', 'parking_space', 
            'empty_space', 'slot'
        ]
        # Check if it's exactly 'ParkingSpot' (case sensitive) or matches other patterns
        if class_name == 'ParkingSpot':
            return True
        return any(cls in class_name.lower() for cls in parking_classes)

    def _create_parking_goal(self, detection):
        """Create a parking goal from YOLO detection"""
        # Convert screen coordinates to world coordinates
        screen_center_x = detection['center'][0]
        screen_center_y = detection['center'][1]
        
        # Convert to world coordinates
        world_x, world_y = self._screen_to_world_coords(screen_center_x, screen_center_y)
        
        print(f"Original parking spot detected at: ({world_x:.2f}, {world_y:.2f})")
        
        # Calculate angle to face center of map
        angle_to_center = self._calculate_angle_to_center(world_x, world_y)
        
        # Determine parking spot orientation and move goal to the FRONT end of the spot (near driving lane)
        # Based on the ParkingLotModule configurations
        if abs(world_x) > abs(world_y):  # Horizontal parking spots (left/right sides)
            # For spots on the left side, goal should be at the LEFT end (away from center)
            # For spots on the right side, goal should be at the RIGHT end (away from center)
            if world_x < 0:  # Left side parking spots
                goal_x = world_x - 1.5  # Move away from center (left end of spot)
            else:  # Right side parking spots  
                goal_x = world_x + 1.5  # Move away from center (right end of spot)
            goal_y = world_y
            print(f"Horizontal spot: moved goal to ({goal_x:.2f}, {goal_y:.2f})")
        else:  # Vertical parking spots (top/bottom)
            # For spots on top, goal at top end (away from center)
            # For spots on bottom, goal at bottom end (away from center)
            if world_y > 0:  # Top parking spots
                goal_y = world_y + 1.5  # Move away from center (top end of spot)
            else:  # Bottom parking spots
                goal_y = world_y - 1.5  # Move away from center (bottom end of spot)
            goal_x = world_x
            print(f"Vertical spot: moved goal to ({goal_x:.2f}, {goal_y:.2f})")
        
        # Create goal object with smaller size
        goal = {
            'position': [goal_x, goal_y],
            'angle': angle_to_center,
            'size': [0.8, 0.8],  # Much smaller goal - car needs to be mostly in the spot
            'confidence': detection['confidence'],
            'class_name': detection['class_name'],
            'bidirectional': True
        }
        
        return goal

    def _screen_to_world_coords(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        if self.current_screen:
            screen_width = self.current_screen.get_width()
            screen_height = self.current_screen.get_height()
            
            # Convert screen coordinates to normalized coordinates (-1 to 1)
            norm_x = (screen_x - screen_width / 2) / (screen_width / 2)
            norm_y = (screen_y - screen_height / 2) / (screen_height / 2)
            
            # Scale to world coordinates - DON'T flip Y axis since pygame and world coords match
            world_x = norm_x * (self.world_width / 2)
            world_y = norm_y * (self.world_height / 2)  # Remove the negative sign
            
            return world_x, world_y
        
        return 0, 0

    def _calculate_angle_to_center(self, x, y):
        """Calculate angle from position to center of map"""
        if x == 0 and y == 0:
            return 0
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = math.atan2(-y, -x)  # Negative to point toward center
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg

    def render(self, screen, transform_matrix):
        # Just update detection - no visual rendering
        self.update_detection(screen)

    def get_digest(self):
        return f"YOLOGoalDetector(goals_count={len(self.parking_goals)})"

    def get_unified_state(self):
        # Convert goals to the format expected by SimulationEnvironment: (x, y, angle)
        formatted_goals = []
        for goal in self.parking_goals:
            formatted_goals.append((
                goal['position'][0], 
                goal['position'][1], 
                math.radians(goal['angle'])  # Convert degrees to radians
            ))
        
        return {
            'name': 'YOLOGoals',
            'goals': formatted_goals,  # Use 'goals' key for compatibility
            'parking_goals': self.parking_goals,  # Keep full goal data
            'detected_objects': self.detected_objects
        }

    def get_parking_goals(self):
        """Get current parking goals for external use"""
        return self.parking_goals

    def toggle_yolo_detection(self):
        """Toggle YOLO detection on/off"""
        return self.yolo_detector.toggle_detection()

    def set_detection_interval(self, interval):
        """Set how often detection runs (in frames)"""
        self.detection_interval = max(1, interval)

    def _get_model_path(self):
        """Get the path to the YOLO model file"""
        import os
        
        # Get the directory where this file is located (modules folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the project root (CarParkingSim)
        project_root = os.path.dirname(current_dir)
        
        # Build path to model
        model_path = os.path.join(project_root, "runs", "detect", "train", "weights", "best.pt")
        
        print(f"Looking for YOLO model at: {model_path}")
        
        if os.path.exists(model_path):
            print(f"YOLO model found!")
            return model_path
        else:
            print(f"YOLO model not found at {model_path}")
            # Try alternative locations
            alternative_paths = [
                os.path.join(project_root, "best.pt"),  # In root directory
                "yolov8n.pt"  # Fallback to default
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Using alternative model: {alt_path}")
                    return alt_path
            
            print("Using default yolov8n.pt model")
            return "yolov8n.pt"