from scripts.replot_training import test_multiple_checkpoints
from math import pi
from modules.parking_modules import parking_env2


def param_converter(**kwargs):
    return kwargs


def main():
    env = parking_env2

    configs = [
        [env, param_converter(goal_size=1.5, decimation=0.95, angle_tolerance=pi / 4, force_first_car=False), 'APYX_STP_1', 100, True],
        [env, param_converter(goal_size=1.5, decimation=0.9, angle_tolerance=pi / 8, force_first_car=False), 'APYX_STP_2', 100, True],
        [env, param_converter(goal_size=1.5, episodes=250, decimation=0.8, force_first_car=False), 'APYX_STP_3', 100, True],
        [env, param_converter(goal_size=1.0, episodes=300, decimation=0.7, force_first_car=False), 'APYX_STP_4', 100, True],
        [env, param_converter(goal_size=1.0, episodes=350, decimation=0.6, force_first_car=False), 'APYX_STP_5', 100, True],
        [env, param_converter(goal_size=0.7, episodes=350, decimation=0.3, angle_tolerance=pi / 10, curb_depth=4, generate_curb=True), 'APYX_STP_6', 100, True],
        [env, param_converter(goal_size=0.7, episodes=200, decimation=0.3, angle_tolerance=pi / 10, generate_curb=True, spawn_probability=0.6), 'APYX_STP_7', 100, True],
        [env, param_converter(goal_size=0.7, episodes=200, decimation=0.3, angle_tolerance=pi / 10, curb_depth=1, generate_curb=True, spawn_probability=0.6, vision=True, vision_model="VIS_006b"), 'APYX_STP_8', 50, True]
    ]

    test_multiple_checkpoints(configs)


if __name__ == "__main__":
    main()
