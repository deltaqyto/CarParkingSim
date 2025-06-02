from Simulation.environments import load_env
from modules.environment_modules import Borders, ParkingLotModule
from modules.reward_functions import GoalEndReward, TimePenalty, CollisionPenalty
from modules.stop_conditions import StepLimit, CollisionStop, GoalStop
from modules.observation_modules import ClassicalObservation
from modules.module_reward_display import RewardDisplayModule
from modules.YOLO_modules import YOLOGoalStop, SmoothDistanceReward, CarProximityPenalty, IncreasingTimePenalty


def get_yolo_env(render=False, goal_size=0.8, angle_tolerance=999, vision_model="DET_001"):
    world_width = 60
    world_aspect = 3 / 4
    world_height = world_width * world_aspect

    # YOLO-enabled environment modulesS
    environment_modules = [
        Borders(),
        ParkingLotModule(configuration=0, generate_goals=True, goal_radius = 0.8),
    ]

    # Use improved YOLO stop condition that handles timing gracefully
    stop_conditions = [
        StepLimit(step_limit=400),
        CollisionStop(),
    ]
    if vision_model is not None:
        stop_conditions.append(YOLOGoalStop(goal_radius=goal_size, model_name=vision_model))
    else:
        stop_conditions.append(GoalStop())

    #YOLO-optimized rewards
    reward_functions = [
        GoalEndReward(reward=100),
        IncreasingTimePenalty(reward=-0.0001),
        CollisionPenalty(reward=-30),
        SmoothDistanceReward(continuous_scale=0.8),
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
        generate_vision=vision_model is not None
    )

    return env
