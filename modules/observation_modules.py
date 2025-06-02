import numpy as np
from os import path
import torch
import glob
from random import random

import matplotlib
import matplotlib.pyplot as plt

from Utility.raycast import Ray, ray_cast

from modules.generic_modules import GenericObservation


class ClassicalObservation(GenericObservation):
    def __init__(self, rescale_lidar=False, lidar_noise=0.0):
        super().__init__()
        self.rescale_lidar = rescale_lidar
        self.lidar_noise = lidar_noise

    def get_observation(self, state, observation):
        rc = state['raycasts']
        rc = [min(1, max(0, r - self.lidar_noise * random())) for r in rc]
        observation = [
            *state['car']['observation'],
            *([n * 2 - 1 for n in rc] if self.rescale_lidar else rc),
            *state['closest_goal']['car_frame'],
        ]
        return observation

    def get_digest(self):
        return f'ClassicalObservation(rescale_lidar={self.rescale_lidar}, lidar_noise={self.lidar_noise})'


class VisionRaycastObservation(GenericObservation):
    def __init__(self, model_name, search_path="models", show_image=False, display_interval=0.1, ray_count=24, expected_image_size=100):
        self.model_name = model_name
        self.search_path = search_path
        self.ray_count = ray_count
        self.expected_image_size = expected_image_size
        self.show_image = show_image
        self.display_interval = display_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = self._load_vision_model()

        # Display setup
        self.fig = None
        self.ax = None
        self.last_display_time = 0

        if self.show_image:
            self._setup_display()

    def _setup_display(self):
        """Setup matplotlib display for debugging."""
        matplotlib.use('TkAgg')
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle(f'Vision Model: {self.model_name}')

        self.ax1.set_title('Preprocessed Image (Model Input)')
        self.ax1.axis('off')

        self.ax2.set_title('Raycast Predictions')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlim(0, 12)
        self.ax2.set_xlabel('Raycast Index')
        self.ax2.set_ylabel('Distance (0=close, 1=far)')
        self.ax2.grid(True, alpha=0.3)

        plt.show(block=False)

    def _load_vision_model(self):
        """Load the best vision model for the given model name."""
        from AI.vision_CNN import RaycastResNet

        # Find vision models matching the name
        model_dir = path.join(self.search_path, "vision_models", f"vision_{self.model_name}")
        if not path.exists(model_dir):
            raise ValueError(f"No vision models found for {self.model_name}")

        # Look for model files
        model_files = glob.glob(path.join(model_dir, "*.pth"))
        if not model_files:
            raise ValueError(f"No .pth files found in {model_dir}")

        # Find the best model (prefer final, then highest batch count)
        best_model_path = None
        best_batch_num = -1

        for model_path in model_files:
            filename = path.basename(model_path)
            if "final" in filename:
                best_model_path = model_path
                break
            elif "_batches.pth" in filename:
                try:
                    # Extract batch number from filename like "VIS_001c_25000_batches.pth"
                    parts = filename.replace('.pth', '').split('_')
                    # Find the part before 'batches'
                    for i, part in enumerate(parts):
                        if part == 'batches' and i > 0:
                            batch_num = int(parts[i - 1])
                            if batch_num > best_batch_num:
                                best_batch_num = batch_num
                                best_model_path = model_path
                            break
                except Exception as e:
                    continue

        if not best_model_path:
            raise ValueError(f"Could not determine best model from {model_files}")

        # print(f"Loading vision model: {best_model_path}")

        # Load the model
        cnn = RaycastResNet(output_dim=self.ray_count).to(self.device)
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        cnn.eval()

        return cnn

    def _preprocess_image(self, image_array, car_pos, car_angle, world_size):
        """Preprocess image like during training."""
        if len(image_array.shape) != 3:
            raise ValueError(f"Expected 3D image array, got shape {image_array.shape}")

        # Use the same crop and rotate function as training
        from AI.vision_train_utils import crop_and_rotate_image

        # This returns a square uint8 image, car-centered and rotated
        image = crop_and_rotate_image(image_array, car_pos, car_angle, world_size, desired_image_size=self.expected_image_size)

        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0

        # Convert to PyTorch format: (channels, height, width)
        image = np.transpose(image, (2, 0, 1))

        return image

    def _update_display(self, image, vision_prediction):
        """Update the display with current image and predictions."""
        import time

        current_time = time.time()
        if current_time - self.last_display_time < self.display_interval:
            return

        try:
            # Show preprocessed image (convert back to HWC for display)
            display_image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            self.ax1.clear()
            self.ax1.imshow(display_image)
            self.ax1.set_title('Preprocessed Image (Model Input)')
            self.ax1.axis('off')

            # Show raycast predictions as bar chart
            self.ax2.clear()
            bars = self.ax2.bar(range(12), vision_prediction, alpha=0.7)

            # Color bars based on distance (red=close, green=far)
            for i, (bar, val) in enumerate(zip(bars, vision_prediction)):
                if val < 0.3:
                    bar.set_color('red')
                elif val < 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')

            self.ax2.set_ylim(0, 1)
            self.ax2.set_xlim(-0.5, 11.5)
            self.ax2.set_xlabel('Raycast Index')
            self.ax2.set_ylabel('Distance (0=close, 1=far)')
            self.ax2.set_title(f'Raycast Predictions (min={vision_prediction.min():.3f}, max={vision_prediction.max():.3f})')
            self.ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, val in enumerate(vision_prediction):
                self.ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.last_display_time = current_time

        except Exception as e:
            print(f"Display error: {e}")

    def get_observation(self, state, observation):
        if len(observation) != 7 + self.ray_count:
            raise ValueError(f"Expected {self.ray_count + 7}-element observation, got {len(observation)}")

        # Get vision prediction (24 elements)
        if 'vision' not in state:
            raise ValueError("No 'vision' key in state - environment must provide vision data")

        # Preprocess image
        image = self._preprocess_image(state['vision'],
                                       state['car']['position'],
                                       np.arctan2(state['car']['direction_vector'][1], state['car']['direction_vector'][0]),
                                       state['world_size'])

        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)

        # Get prediction from CNN
        with torch.no_grad():
            prediction = self.vision_model(image_tensor)
            vision_prediction = prediction.cpu().numpy().flatten()

        if len(vision_prediction) != self.ray_count:
            raise ValueError(f"Expected ray_count-element vision prediction, got {len(vision_prediction)}")

        # Update display if enabled
        if self.show_image and self.fig is not None:
            self._update_display(image, vision_prediction)

        state['raycasts'] = vision_prediction
        observation[4:4 + self.ray_count] = vision_prediction

        return observation

    def get_digest(self):
        return f'VisionObservation(model_name={self.model_name}, show_image={self.show_image}, ray_count={self.ray_count}, expected_image_size={self.expected_image_size})'

    def close(self):
        """Clean up display resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
