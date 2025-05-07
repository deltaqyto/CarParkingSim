from Simulation.simulation_environment import SimulationEnvironment

# Yes, these might show as import errors. It resolves at runtime
from modules.environment_modules import Borders
from modules.reward_functions import GoalEndReward, TimePenalty, CollisionPenalty, DistanceReward
from modules.stop_conditions import bidirectional_goal, StepLimit, CollisionStop

# Makes a unique environment for each thread
def load_env(**kwargs):
    def func():
        return SimulationEnvironment(**kwargs)
    return func

# Example of how to make an environment with training functions. Make more if you want
def get_basic_env(render=False, goal_size=1, angle_tolerance=1):
    world_width = 60
    world_aspect = 3 / 4
    world_height = world_width * world_aspect

    # Register your modules here
    # Environment modules can draw rectangles to the screen (or other items). They execute first in the chain
    environment_modules = [Borders()]

    # Stop conditions run second. They also get to render to the screen, and decide whether to stop the current episode
    # There is no safety mechanism if you forget to add a step limit. Always ensure the environment will stop
    stop_conditions = [StepLimit(step_limit=200),
                       CollisionStop(),
                       bidirectional_goal(region=[-world_width / 2 * 0.8, world_width / 2 * 0.8, -world_height / 2 * 0.8, world_height / 2 * 0.8], goal_size=goal_size, angle_tolerance=angle_tolerance)]

    # Reward functions run last. These are provided with the environment state, and decide on a reward. Rewards are summed and output.
    # Reward values are stored individually for your convenience. See the state output of env.step()
    reward_functions = [GoalEndReward(),
                        TimePenalty(),
                        CollisionPenalty(),
                        DistanceReward()]

    env = load_env(render=render, world_width=world_width, world_aspect=world_aspect,
                   stop_conditions=stop_conditions, environment_modules=environment_modules, reward_functions=reward_functions)
    # Parameters:
    # render=False, console_logger=None, discrete=False, screen_width=800,
    # stop_conditions=None, car_params=None):

    return env
