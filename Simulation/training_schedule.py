from math import pi
from multiprocessing import cpu_count
from Simulation.environments import get_basic_env

class GenericTrainingSchedule:
    def __init__(self, start_from_env=0):
        self.environments = []
        self.parameters = []
        self.start_from_env = start_from_env
        self.current_environment = start_from_env

    def get_next_environment(self):
        env = self.environments[self.current_environment]

        populated_params = {}
        for i in range(self.current_environment + 1):
            if i > len(self.parameters) - 1:
                break
            populated_params = {**populated_params, **self.parameters[i]}

        self.current_environment += 1
        return env, populated_params

    def get_nth_environment(self, env_number):
        env_number = env_number + self.start_from_env
        env = self.environments[env_number]

        populated_params = {}
        for i in range(env_number + 1):
            if i > len(self.parameters) - 1:
                break
            populated_params = {**populated_params, **self.parameters[i]}

        return env, populated_params

    def get_num_environments(self):
        return len(self.environments) - self.start_from_env

    def get_digest(self):
        env_strings = [env().get_digest() for env in self.environments]
        param_strings = [str(param) for param in self.parameters]
        return f"TrainingSchedule('{self.__class__.__name__}')[envs:\n" + '\n'.join(env_strings) + '\nparams:\n' + '\n'.join(param_strings) + '\n]'


# Make your training schedule here. Use this as a template to help
class BasicTrainingSchedule(GenericTrainingSchedule):
    def __init__(self):
        super().__init__()
        # The trainer will execute each of these environments in order provided.
        # You can use this to make progressively harder tasks
        self.environments = [get_basic_env(goal_size=2, angle_tolerance=pi/4), get_basic_env(goal_size=1, angle_tolerance=pi/8), get_basic_env(goal_size=0.5, angle_tolerance=pi/16)]

        # You can customise training parameters for each environment. If you leave out a value, it will be copied from the previous environment.
        # These are piped directly into the training algorithm
        base_params = {
            'num_envs': min(14, max(1, cpu_count())),
            'action_dim': 2,  # throttle, steering
            'batch_size': 256,
            'total_timesteps': 3_000_000,
            'save_freq': 20000,
            'eval_episodes': 10,
            'seed': 41,
            'exploration_noise': 0.1,
            'start_timesteps': 25000,  # Random exploration steps
            'buffer_size': 1_000_000,
            'learning_rate': 3e-4,
            'net_size': [400, 300],
        }
        self.parameters = [base_params, {'total_timesteps': 1_500_000}]  # The second and third environments run on half the timesteps as the first run, all else is the same


class YOLOParkingSchedule(GenericTrainingSchedule):
    def __init__(self, render=False):
        super().__init__()
        # Import YOLO environment
        from modules.custom_environments import get_yolo_env
        
        # Progressive YOLO parking difficulty - adjust goal radius for curriculum learning
        self.environments = [
            get_yolo_env(goal_size=2.5, angle_tolerance=pi/2, render=render)   # Stage 1: Large goal, any angle
            #get_yolo_env(goal_size=2.0, angle_tolerance=pi/4, render=render),   # Stage 2: Medium goal, loose angle
            #get_yolo_env(goal_size=1.5, angle_tolerance=pi/6, render=render),   # Stage 3: Smaller goal, tighter angle
            #get_yolo_env(goal_size=1.0, angle_tolerance=pi/8, render=render)    # Stage 4: Precise parking
        ]

        # YOLO-optimized training parameters
        base_params = {
            'num_envs': min(3, max(1, 3)),  # Slightly more envs for YOLO
            'action_dim': 2,  # throttle, steering
            'batch_size': 256,
            'total_timesteps': 4_000_000,  # More timesteps for YOLO navigation
            'save_freq': 25000,
            'eval_episodes': 15,  # More evaluation episodes
            'seed': 42,
            'exploration_noise': 0.15,  # Higher exploration for parking
            'start_timesteps': 40000,  # More random exploration for YOLO environment
            'buffer_size': 1_500_000,  # Larger buffer for complex parking scenarios
            'learning_rate': 2e-4,  # Slightly lower LR for stability
            'net_size': [512, 256],  # Larger network for complex navigation
        }
        
        # Progressive parameter adjustments for each stage
        self.parameters = [
            base_params,  # Stage 1: Full exploration
            {
                'total_timesteps': 3_500_000,
                'exploration_noise': 0.12,
                'start_timesteps': 35000
            },  # Stage 2: Reduce exploration slightly
            {
                'total_timesteps': 3_000_000,
                'exploration_noise': 0.1,
                'start_timesteps': 30000,
                'learning_rate': 1.5e-4
            },  # Stage 3: Further reduce exploration, lower LR
            {
                'total_timesteps': 2_500_000,
                'exploration_noise': 0.08,
                'start_timesteps': 25000,
                'learning_rate': 1e-4
            }   # Stage 4: Fine-tuning phase
        ]
