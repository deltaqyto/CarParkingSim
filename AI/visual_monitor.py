import os
import random
import time
import threading
import glob
import multiprocessing as mp
from stable_baselines3 import TD3

from AI.train_utils import get_timestep_from_path, get_best_model
from Simulation.simulation_wrapper import SimulationWrapper


def environment_worker(schedule_class, schedule_kwargs, env_index, model_queue, control_queue):
    """Worker process that runs the environment with pygame isolation."""

    schedule = schedule_class(**schedule_kwargs)
    env_factory = schedule.get_nth_environment(env_index)[0]
    env = SimulationWrapper(env_factory, 0, 42)
    current_model = None

    while True:
        try:
            # Check for new model
            if not model_queue.empty():
                model_path = model_queue.get()
                if model_path is None:  # Shutdown signal
                    break
                try:
                    current_model = TD3.load(model_path)
                    print(f"Worker loaded model: {os.path.basename(model_path)}")
                    # Reset environment when new model is loaded
                    env.reset()
                except Exception as e:
                    print(f"Worker failed to load model {model_path}: {e}")
                    current_model = None

            # Check for control commands
            if not control_queue.empty():
                command = control_queue.get()
                if command == 'shutdown':
                    break
                elif command == 'reset_env':
                    # Create new environment (for target switching)
                    env.close()
                    schedule = schedule_class(**schedule_kwargs)
                    env_factory = schedule.get_nth_environment(env_index)[0]
                    env = SimulationWrapper(env_factory, 0, random.randint(1, 100000))

            # Run episode if we have a model
            if current_model is not None:
                observation, _ = env.reset()
                done = False

                while not done:
                    # Get action from model
                    action, _ = current_model.predict(observation, deterministic=True)

                    # Step environment
                    observation, _, done, _, _ = env.step(action)
            else:
                # No model, just wait
                time.sleep(0.1)

        except Exception as e:
            print(f"Error in environment worker: {e}")
            time.sleep(1.0)

    # Cleanup
    env.close()


class VisualMonitor:
    """Visual monitor that runs the latest available model in a subprocess
    """

    def __init__(self, schedule_class, schedule_kwargs, model_targets, models_path="models", check_interval=2.0):
        """
        Args:
            schedule_class: Schedule class (e.g., ParkingSchedule2)
            schedule_kwargs: Kwargs for schedule creation (e.g., {'render': True})
            model_targets: Dict of {'model_name': env_index}
            models_path: Path to models directory
            check_interval: How often to check for model updates (seconds)
        """
        self.schedule_class = schedule_class
        self.schedule_kwargs = schedule_kwargs
        self.model_targets = model_targets
        self.models_path = models_path
        self.check_interval = check_interval

        # Current state
        self.current_model_path = None
        self.current_target = None

        # Process communication
        self.model_queue = None
        self.control_queue = None
        self.worker_process = None

        # Control flags
        self.running = True
        self.monitor_thread = None
        self.waiting_logged = False

    def run(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        try:
            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            self._cleanup()

    def _monitor_loop(self):
        """Background monitoring for model updates."""
        while self.running:
            try:
                # Find best available model
                target_name, model_path = self._find_best_model()

                if model_path and model_path != self.current_model_path:
                    print(f"New model detected: {target_name} -> {os.path.basename(model_path)}")
                    self.waiting_logged = False

                    # Check if we're switching targets (need new environment)
                    if target_name != self.current_target:
                        print(f"Switching to new target: {target_name}")
                        self._switch_target(target_name, model_path)
                    else:
                        # Same target, just send new model
                        if self.model_queue:
                            self.model_queue.put(model_path)
                            self.current_model_path = model_path

                elif not model_path and not self.waiting_logged:
                    print("Waiting for model to load...")
                    self.waiting_logged = True

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(self.check_interval)

    def _find_best_model(self):
        # Check targets in reverse order (last in dict has priority)
        for target_name in reversed(list(self.model_targets.keys())):
            model_dir = os.path.join(self.models_path, f"td3_{target_name}")

            if os.path.exists(model_dir):
                model_files = []

                final_pattern = os.path.join(model_dir, f"{target_name}_final.zip")
                final_models = glob.glob(final_pattern)
                model_files.extend(final_models)

                step_pattern = os.path.join(model_dir, f"{target_name}_*_steps.zip")
                step_models = glob.glob(step_pattern)
                model_files.extend(step_models)

                if model_files:
                    best_model = get_best_model(model_files)
                    if best_model:
                        return target_name, best_model

        return None, None

    def _switch_target(self, target_name, model_path):
        """Switch to a new target (different environment)."""
        self._cleanup()

        self.model_queue = mp.Queue()
        self.control_queue = mp.Queue()

        env_index = self.model_targets[target_name]

        # Start new worker process
        self.worker_process = mp.Process(
            target=environment_worker,
            args=(self.schedule_class, self.schedule_kwargs, env_index, self.model_queue, self.control_queue)
        )
        self.worker_process.start()

        # Send initial model
        self.model_queue.put(model_path)
        self.current_model_path = model_path
        self.current_target = target_name

    def _cleanup(self):
        """Clean up worker process."""
        if self.worker_process and self.worker_process.is_alive():
            # Send shutdown signals
            if self.model_queue:
                self.model_queue.put(None)
            if self.control_queue:
                self.control_queue.put('shutdown')

            # Wait for process to finish
            self.worker_process.join(timeout=5.0)
            if self.worker_process.is_alive():
                self.worker_process.terminate()
                self.worker_process.join()

        self.worker_process = None
        self.model_queue = None
        self.control_queue = None
