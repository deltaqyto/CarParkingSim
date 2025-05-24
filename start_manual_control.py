import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from os import path
import sys
import cv2

from Objects.car import Car
from Utility.console_logger import ConsoleLogger
from Simulation.simulation_environment import SimulationEnvironment
from AI.yolo_detector import YOLODetector  # Import the YOLODetector class

from pyinstrument import Profiler # pip install pygame numpy pyinstrument

def capture_pygame_screen(screen):
    """
    Capture the current Pygame screen as a numpy array
    
    Args:
        screen (pygame.Surface): Pygame screen surface
    
    Returns:
        numpy.ndarray: Captured screen as a numpy array in BGR format
    """
    # Convert Pygame surface to numpy array
    screen_array = pygame.surfarray.array3d(screen)
    
    # Transpose to get standard image format (Height, Width, Channels)
    screen_array = screen_array.transpose([1, 0, 2])
    
    # Convert from RGB to BGR (OpenCV default)
    screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    
    return screen_array

def draw_detections(screen, detections, transform):
    """
    Draw YOLO detections on Pygame screen
    
    Args:
        screen (pygame.Surface): Pygame screen surface
        detections (list): List of detection dictionaries
        transform (numpy.ndarray): Transformation matrix
    """
    for detection in detections:
        # Extract bounding box
        x1, y1, x2, y2 = map(int, detection['bbox'])
        
        # Choose color based on confidence
        color = (0, 255, 0)  # Green for high confidence
        if detection['confidence'] < 0.5:
            color = (255, 255, 0)  # Yellow for low confidence
        
        # Draw rectangle
        pygame.draw.rect(screen, color, (x1, y1, x2-x1, y2-y1), 2)
        
        # Render class name and confidence
        font = pygame.font.SysFont(None, 24)
        text = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x1, y1-30))

def process_detections(detections, sim_env):
    """
    Process detections and potentially interact with simulation environment
    
    Args:
        detections (list): List of detection dictionaries
        sim_env (SimulationEnvironment): Simulation environment instance
    
    Returns:
        list: Processed detection information
    """
    processed_objects = []
    for detection in detections:
        # Convert bbox to world coordinates
        x1, y1, x2, y2 = map(int, detection['bbox'])
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Convert screen coordinates to world coordinates
        # This might require using sim_env.transform or a custom transformation
        # The exact implementation depends on your coordinate system
        try:
            # Attempt to transform screen coordinates to world coordinates
            # This is a placeholder and should be replaced with your specific transformation logic
            world_coords = np.linalg.inv(sim_env.transform).dot(np.array([center_x, center_y, 1]))
            
            processed_objects.append({
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'world_x': world_coords[0],
                'world_y': world_coords[1]
            })
        except Exception as e:
            print(f"Error processing detection: {e}")
    
    return processed_objects

def main():
    # Initialize YOLO Detector
    detector = YOLODetector()
    
    # Optional: If you want to train or convert dataset first
    # Uncomment and modify as needed:
    # detector.convert_label_studio_to_yolo(
    #     input_dir='/path/to/label/studio/annotations',
    #     output_dir='/path/to/yolo/dataset',
    #     train_ratio=0.8  # 80% train, 20% validation
    # )
    # detector.train(
    #     epochs=50,
    #     imgsz=640,
    #     batch=16
    # )

    render = True
    instrument = False
    sim_env = SimulationEnvironment(render=render)
    print('=' * 20 + " Digest " + '=' * 20)
    print(sim_env.get_digest())
    print('=' * 20 + " End Digest " + '=' * 20)
    rewards = 0

    # Create profiler
    if instrument:
        profiler = Profiler()
        profiler.start()

    # Number of steps to run
    num_steps = 10000
    step_count = 0

    # Detection frequency control
    detection_interval = 10  # Process detection every 10 steps
    
    # Store detected objects
    detected_objects = []

    while True:
        if instrument and step_count >= num_steps:
            break

        throttle = 0
        steer = 0
        if render and not instrument:
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_DOWN]:
                throttle = -1.0
            if keys[pygame.K_LEFT]:
                steer = -1.0
            if keys[pygame.K_RIGHT]:
                steer = 1.0
            if keys[pygame.K_r]:
                sim_env.reset_environment()
                throttle = 0
                steer = 0
            if keys[pygame.K_q]:
                break

        done, observation, reward, state = sim_env.step([throttle, steer])
        rewards += reward
        step_count += 1

        # Perform object detection periodically to reduce computational load
        if render and not instrument and step_count % detection_interval == 0:
            # Capture current screen
            current_screen = capture_pygame_screen(sim_env.screen)
            
            # Perform detection
            detections = detector.detect(current_screen)
            
            # Process detections
            detected_objects = process_detections(detections, sim_env)
            
            # Draw detections on screen
            draw_detections(sim_env.screen, detections, sim_env.transform)
            
            # Update display to show detections
            pygame.display.flip()
            
            # Optional: Log or use detected objects
            for obj in detected_objects:
                print(f"Detected {obj['class_name']} at ({obj['world_x']}, {obj['world_y']}) with confidence {obj['confidence']}")

        if done:
            print(f"Episode reward: {rewards}")
            rewards = 0
            sim_env.reset_environment()

        if 'User Quit' in state['stop_reasons']:
            break

    # Stop profiling and print results
    if instrument:
        profiler.stop()

        # Print to console
        print(profiler.output_text(unicode=True, color=True))

        # Generate HTML report
        profiler.write_html("profile_report.html")
        print("Detailed HTML profile saved to 'profile_report.html'")

if __name__ == "__main__":
    main()