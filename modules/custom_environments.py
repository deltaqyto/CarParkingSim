from Simulation.environments import load_env
from modules.environment_modules import Borders, ParkingLotModule
from modules.reward_functions import GoalEndReward, TimePenalty, SmoothCollisionPenalty, SmoothDistanceReward, CarProximityPenalty
from modules.stop_conditions import StepLimit, CollisionStop
from modules.observation_modules import ClassicalObservation
from modules.module_reward_display import RewardDisplayModule
from modules.YOLO_modules import YOLOGoalDetector, YOLOGoalStop


def get_yolo_env(render=False, goal_size=1, angle_tolerance=1, vision=True):
    world_width = 60
    world_aspect = 3 / 4

    # YOLO-enabled environment modulesS
    environment_modules = [
        Borders(),
        ParkingLotModule(configuration=0),
        YOLOGoalDetector(model_name="DET_001")
    ]

    # Use improved YOLO stop condition that handles timing gracefully
    stop_conditions = [
        StepLimit(step_limit=400),
        CollisionStop(),
        YOLOGoalStop(goal_radius=0.8)  # Use improved version
    ]

    # YOLO-optimized rewards
    reward_functions = [
        GoalEndReward(reward=100),
        TimePenalty(reward=-0.01),
        SmoothCollisionPenalty(reward=-15, car_penalty_multiplier=2.0),
        SmoothDistanceReward(continuous=True, continuous_scale=0.8),
        CarProximityPenalty(penalty_distance=2.5, max_penalty=-0.03, exploration_bonus=0.005),
        RewardDisplayModule()

    ]

    env = load_env(
        render=render,
        world_width=world_width,
        world_aspect=world_aspect,
        stop_conditions=stop_conditions,
        environment_modules=environment_modules,
        reward_functions=reward_functions,
        observation_modules=[ClassicalObservation()],
        generate_vision=vision
    )

    return env
