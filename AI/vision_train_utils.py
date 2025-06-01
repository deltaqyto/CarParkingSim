import os
import time
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from Simulation.simulation_wrapper import SimulationWrapper
from AI.train_utils import find_td3_models, get_best_model
from AI.vision_CNN import VisionCNN, WeightedRaycastLoss, RaycastResNet


class ThreadSafeBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, image, observation):
        with self.lock:
            self.buffer.append((image, observation))

    def sample_batch(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None, None

            samples = random.sample(list(self.buffer), batch_size)
            images, observations = zip(*samples)
            return np.array(images), np.array(observations)

    def size(self):
        with self.lock:
            return len(self.buffer)

    def is_ready(self, min_size):
        return self.size() >= min_size


def crop_and_rotate_image(image_array, car_pos, car_angle, world_size, desired_image_size):
    """Crop around car and rotate so car faces same direction."""
    # Configurable crop sizes
    FINAL_CROP_SIZE = desired_image_size
    LARGE_CROP_SIZE = round(FINAL_CROP_SIZE * 1.5)

    if len(image_array.shape) != 3:
        raise ValueError(f"Expected 3D image array, got shape {image_array.shape}")

    # Convert from pygame format: (width, height, 3) -> (height, width, 3)
    image = np.transpose(image_array, (1, 0, 2))

    # Downscale to 400x300 first
    image = cv2.resize(image, (400, 300))

    # Convert world coordinates to image coordinates
    world_width, world_height = world_size
    image_width, image_height = 400, 300

    # Calculate scale factors
    scale_x = image_width / world_width
    scale_y = image_height / world_height

    # Convert car position from world coordinates to image coordinates
    car_x = int(car_pos[0] * scale_x + image_width / 2)
    car_y = int(car_pos[1] * scale_y + image_height / 2)

    # Do larger crop first to account for rotation
    large_crop_half = LARGE_CROP_SIZE // 2

    # Calculate crop bounds for larger crop
    x_start = car_x - large_crop_half
    x_end = car_x + large_crop_half
    y_start = car_y - large_crop_half
    y_end = car_y + large_crop_half

    # Create black canvas for larger crop
    large_cropped = np.zeros((LARGE_CROP_SIZE, LARGE_CROP_SIZE, 3), dtype=np.uint8)

    # Calculate valid regions for copying
    src_x_start = max(0, x_start)
    src_x_end = min(image_width, x_end)
    src_y_start = max(0, y_start)
    src_y_end = min(image_height, y_end)

    dst_x_start = max(0, -x_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_start = max(0, -y_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # Copy valid region to black canvas
    if src_x_end > src_x_start and src_y_end > src_y_start:
        large_cropped[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]

    # Rotate the larger crop so car always faces up
    rotation_angle = np.degrees(car_angle)
    center = (large_crop_half, large_crop_half)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Apply rotation to larger crop
    rotated_large = cv2.warpAffine(large_cropped, M, (LARGE_CROP_SIZE, LARGE_CROP_SIZE))

    # Final crop from center of rotated image
    final_crop_start = (LARGE_CROP_SIZE - FINAL_CROP_SIZE) // 2
    final_crop_end = final_crop_start + FINAL_CROP_SIZE

    final_cropped = rotated_large[final_crop_start:final_crop_end, final_crop_start:final_crop_end]

    return final_cropped.astype(np.uint8)


def extract_raycasts(observation, output_rays):
    """Extract raycast values from observation (indices 4:output_rays + 4)"""
    obs = np.array(observation)
    if len(obs) < output_rays + 4:
        raise ValueError(f"Expected at minimum {output_rays + 4}-element observation, got {len(obs)} elements")

    # Extract raycasts (start at index 4, take n rays)
    raycasts = obs[4:output_rays + 4]
    return raycasts.astype(np.float32)


class DisplayThread(threading.Thread):
    def __init__(self, buffer, display_interval, training_image_size):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.display_interval = display_interval
        self.training_image_size = training_image_size
        self.running = True
        self.fig = None
        self.ax = None

    def run(self):
        try:
            # Set backend for display
            matplotlib.use('TkAgg')
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))  # Square images
            self.fig.suptitle(f'Random Training Image (Car-Centric {self.training_image_size}x{self.training_image_size})')
            plt.show(block=False)

            while self.running:
                if self.buffer.size() > 0:
                    with self.buffer.lock:
                        if len(self.buffer.buffer) > 0:
                            image, obs = random.choice(list(self.buffer.buffer))

                            self.ax.clear()
                            self.ax.imshow(image)
                            self.ax.set_title(f'Buffer size: {len(self.buffer.buffer)}')
                            self.ax.axis('off')

                            self.fig.canvas.draw()
                            self.fig.canvas.flush_events()

                time.sleep(self.display_interval)

        except Exception as e:
            print(f"Display thread error: {e}")

    def stop(self):
        self.running = False
        if self.fig:
            plt.close(self.fig)


class DataCollectionThread(threading.Thread):
    def __init__(self, environment_factory, rl_model, buffer, thread_id, training_image_size, output_rays, seed_offset=0):
        super().__init__(daemon=True)
        self.environment_factory = environment_factory
        self.rl_model = rl_model
        self.buffer = buffer
        self.thread_id = thread_id
        self.seed_offset = seed_offset
        self.training_image_size = training_image_size
        self.output_rays = output_rays
        self.running = True
        self.stats = {'episodes': 0, 'steps': 0, 'errors': 0}

    def run(self):
        try:
            env = SimulationWrapper(self.environment_factory, self.thread_id, 42 + self.seed_offset)

            while self.running:
                try:
                    observation, _ = env.reset()
                    done = False

                    while not done and self.running:
                        action, _ = self.rl_model.predict(observation, deterministic=True)
                        observation, reward, done, truncated, state = env.step(action)

                        if state is None:
                            raise ValueError("Environment returned None state")

                        if 'vision' not in state or state['vision'] is None:
                            raise ValueError("No 'vision' key in environment state")

                        if 'world_size' not in state:
                            raise ValueError("No 'world_size' key in environment state")

                        # Get car position and orientation
                        car_pos = state['car']['position']
                        car_direction = state['car']['direction_vector']
                        car_angle = np.arctan2(car_direction[1], car_direction[0])
                        world_size = state['world_size']

                        # Process image and observation
                        image = crop_and_rotate_image(state['vision'], car_pos, car_angle, world_size, self.training_image_size)
                        raycasts = extract_raycasts(observation, output_rays=self.output_rays)

                        self.buffer.add(image, raycasts)
                        self.stats['steps'] += 1

                        done = done or truncated

                    self.stats['episodes'] += 1

                except Exception as e:
                    self.stats['errors'] += 1
                    print(f"Collection thread {self.thread_id} error: {e}")
                    raise e

        except Exception as e:
            print(f"Fatal error in collection thread {self.thread_id}: {e}")
            raise e

    def stop(self):
        self.running = False


class VisionTrainer:
    def __init__(self, environment_factory, vision_params, train_id, search_path="models"):
        self.environment_factory = environment_factory
        self.params = vision_params
        self.train_id = train_id
        self.search_path = search_path

        self.model_dir = os.path.join(search_path, "vision_models", f"vision_{train_id}")
        os.makedirs(self.model_dir, exist_ok=True)

        self.rl_model = self._load_rl_model()

        # Set max_batches with default
        self.max_batches = vision_params.get('max_batches', 500)
        self.training_image_size = vision_params.get('final_image_size', 100)
        self.output_rays = vision_params.get('output_rays', 24)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = RaycastResNet(input_size=self.training_image_size, output_dim=self.output_rays).to(self.device)
        self.optimizer = torch.optim.Adam(self.vision_model.parameters(), lr=vision_params['learning_rate'])
        self.criterion = WeightedRaycastLoss(
            weight_far=0.3,  # Lower weight for raycasts = 1.0 (no obstacles)
            weight_near=3.0,  # Higher weight for raycasts < 0.9 (obstacles detected)
            threshold=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500
        )

        self.buffer = ThreadSafeBuffer(vision_params['vision_buffer_size'])
        self.collection_threads = []
        self.display_thread = None
        self.training_stats = {'batches': 0, 'total_loss': 0.0, 'recent_losses': deque(maxlen=100)}
        self.plot_data = {'batches': [], 'losses': [], 'recent_avgs': []}
        self.running = True

        print(f"Vision trainer initialized for {train_id}")
        print(f"Model directory: {self.model_dir}")
        print(f"Device: {self.device}")
        print(f"Buffer size: {vision_params['vision_buffer_size']}")
        print(f"Collection threads: {vision_params['vision_collection_threads']}")
        print(f"Max batches: {self.max_batches}")

    def _load_rl_model(self):
        rl_model_name = self.params['rl_model_name']
        models = find_td3_models(self.search_path, rl_model_name)

        if not models:
            raise ValueError(f"No models found for {rl_model_name}")

        best_model_path = get_best_model(models)
        if not best_model_path:
            raise ValueError(f"Could not determine best model for {rl_model_name}")

        print(f"Loading RL model: {best_model_path}")
        return TD3.load(best_model_path)

    def start_collection_threads(self):
        num_threads = self.params['vision_collection_threads']

        for i in range(num_threads):
            thread = DataCollectionThread(
                environment_factory=self.environment_factory,
                rl_model=self.rl_model,
                buffer=self.buffer,
                thread_id=i,
                training_image_size=self.training_image_size,
                output_rays=self.output_rays,
                seed_offset=i * 1000
            )
            thread.start()
            self.collection_threads.append(thread)

        print(f"Started {num_threads} collection threads")

    def wait_for_initial_data(self, min_samples=None):
        if min_samples is None:
            min_samples = self.params['batch_size'] * 4

        print(f"Waiting for initial data collection ({min_samples} samples)...")
        while not self.buffer.is_ready(min_samples) and self.running:
            current_size = self.buffer.size()
            print(f"Buffer: {current_size}/{min_samples} samples")
            time.sleep(2.0)

        print(f"Initial data collection complete: {self.buffer.size()} samples")

    def train_batch(self):
        batch_size = self.params['batch_size']
        images, observations = self.buffer.sample_batch(batch_size)

        if images is None:
            return None

        # Track balancing stats
        original_obstacle_ratio = np.mean(np.sum(observations < 0.9, axis=1) > 0)

        obstacle_count = np.sum(observations < 0.9, axis=1)
        balanced = False
        if np.random.random() < 0.3 and np.any(obstacle_count > 0):
            obstacle_indices = np.where(obstacle_count > 0)[0]
            if len(obstacle_indices) > batch_size // 2:
                selected = np.random.choice(obstacle_indices, batch_size // 2, replace=False)
                remaining = np.random.choice(len(observations), batch_size - len(selected), replace=False)
                indices = np.concatenate([selected, remaining])
                images = images[indices]
                observations = observations[indices]
                balanced = True

        images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
        images = images.to(self.device)
        observations = torch.FloatTensor(observations).to(self.device)

        self.optimizer.zero_grad()
        predictions = self.vision_model(images)
        loss = self.criterion(predictions, observations)

        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()

        final_obstacle_ratio = torch.mean((torch.sum(observations < 0.9, dim=1) > 0).float()).item()

        self.training_stats['batches'] += 1
        self.training_stats['total_loss'] += loss_value
        self.training_stats['recent_losses'].append(loss_value)

        # Store balancing info
        if not hasattr(self.training_stats, 'balance_info'):
            self.training_stats['balance_info'] = []

        self.training_stats['balance_info'].append({
            'balanced': balanced,
            'obstacle_ratio': final_obstacle_ratio
        })

        return loss_value

    def save_checkpoint(self):
        batch_num = self.training_stats['batches']
        checkpoint_path = os.path.join(self.model_dir, f"{self.train_id}_{batch_num}_batches.pth")

        torch.save({
            'model_state_dict': self.vision_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_num': batch_num,
            'training_stats': self.training_stats
        }, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")

    def train(self):
        try:
            self.start_collection_threads()

            if self.params.get('show_random', 0) > 0:
                self.display_thread = DisplayThread(self.buffer, self.params['show_random'], self.training_image_size)
                self.display_thread.start()
                print(f"Started display thread (interval: {self.params['show_random']}s)")

            self.wait_for_initial_data()

            save_freq = self.params['save_freq']
            last_print = time.time()

            print("Starting vision training...")

            while self.running and self.training_stats['batches'] < self.max_batches:
                loss = self.train_batch()

                if loss is None:
                    time.sleep(0.1)
                    continue

                if time.time() - last_print > 10.0:
                    recent_loss = np.mean(list(self.training_stats['recent_losses'])) if self.training_stats['recent_losses'] else 0
                    buffer_size = self.buffer.size()

                    # Calculate recent obstacle ratio from stored stats
                    if 'balance_info' in self.training_stats and self.training_stats['balance_info']:
                        recent_balance = self.training_stats['balance_info'][-100:]
                        avg_obstacle_ratio = np.mean([info['obstacle_ratio'] for info in recent_balance])
                        balance_rate = np.mean([info['balanced'] for info in recent_balance])

                        print(f"Batch {self.training_stats['batches']}/{self.max_batches}: Loss {loss:.6f}, Recent Avg {recent_loss:.6f}, Buffer {buffer_size}")
                        print(f"  Obstacle ratio: {avg_obstacle_ratio:.3f}, Balance rate: {balance_rate:.3f}")
                    else:
                        print(f"Batch {self.training_stats['batches']}/{self.max_batches}: Loss {loss:.6f}, Recent Avg {recent_loss:.6f}, Buffer {buffer_size}")

                    last_print = time.time()

                if self.training_stats['batches'] % save_freq == 0:
                    self.save_checkpoint()

            if self.training_stats['batches'] >= self.max_batches:
                print(f"Training completed: reached max batches ({self.max_batches})")

        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            self.stop()

    def stop(self):
        print("Stopping vision training...")
        self.running = False

        for thread in self.collection_threads:
            thread.stop()

        if self.display_thread:
            self.display_thread.stop()

        for thread in self.collection_threads:
            thread.join(timeout=2.0)

        if self.display_thread:
            self.display_thread.join(timeout=1.0)

        final_path = os.path.join(self.model_dir, f"{self.train_id}_final.pth")
        torch.save({
            'model_state_dict': self.vision_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_num': self.training_stats['batches'],
            'training_stats': self.training_stats
        }, final_path)
        print(f"Final model saved to {final_path}")

        for i, thread in enumerate(self.collection_threads):
            stats = thread.stats
            print(f"Collection thread {i}: {stats['episodes']} episodes, {stats['steps']} steps, {stats['errors']} errors")


def do_vision_training(environment_factory, vision_params, train_id, search_path="models"):
    required_params = ['rl_model_name', 'vision_buffer_size', 'vision_collection_threads',
                       'batch_size', 'learning_rate', 'save_freq']

    for param in required_params:
        if param not in vision_params:
            raise ValueError(f"Missing required vision parameter: {param}")

    trainer = VisionTrainer(environment_factory, vision_params, train_id, search_path)
    trainer.train()

    return trainer
