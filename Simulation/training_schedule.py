from math import pi
from multiprocessing import cpu_count
from Simulation.environments import get_basic_env

class GenericTrainingSchedule:
    def __init__(self):
        self.environments = []
        self.parameters = []
        self.current_environment = 0

        self.last_params = {}

    def get_next_environment(self):
        env = self.environments[self.current_environment]
        params = self.parameters[self.current_environment] if self.current_environment < len(self.parameters) else {}

        populated_params = {**self.last_params, **params}  # Update with previous parameters
        self.last_params = populated_params

        self.current_environment += 1
        return env, populated_params

    def get_num_environments(self):
        return len(self.environments)

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
