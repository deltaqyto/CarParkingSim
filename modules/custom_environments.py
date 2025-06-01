from Simulation.environments import load_env
from modules.environment_modules import Borders, ParkingLotModule
from modules.reward_functions import GoalEndReward, TimePenalty, SmoothCollisionPenalty, SmoothDistanceReward, CarProximityPenalty
from modules.stop_conditions import StepLimit, CollisionStop, bidirectional_goal
from modules.observation_modules import ClassicalObservation
from modules.module_reward_display import RewardDisplayModule
from modules.YOLO_modules import YOLOGoalStop


def get_yolo_env(render=False, goal_size=0.8, angle_tolerance=999, vision_model="DET_001"):
    world_width = 60
    world_aspect = 3 / 4
    world_height = world_width * world_aspect

    # YOLO-enabled environment modulesS
    environment_modules = [
        Borders(),
        ParkingLotModule(configuration=0),
    ]

    # Use improved YOLO stop condition that handles timing gracefully
    stop_conditions = [
        StepLimit(step_limit=400),
        CollisionStop(),
    ]
    if vision_model is not None:
        stop_conditions.append(YOLOGoalStop(goal_radius=goal_size, model_name=vision_model))
    else:
        stop_conditions.append(bidirectional_goal(region=[-world_width / 2 * 0.8, world_width / 2 * 0.8, -world_height / 2 * 0.8, world_height / 2 * 0.8], goal_size=goal_size, angle_tolerance=angle_tolerance))

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
        generate_vision=vision_model is not None
    )

    return env
