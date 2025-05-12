from scripts.replot_training import test_model_interactive, test_multiple_checkpoints
from math import pi
from Simulation.environments import get_basic_env


def main():
    env = get_basic_env
    params = {"goal_size":2, "angle_tolerance":pi / 4}
    configs=['1']

    if not configs:
        test_model_interactive(env, params)
    else:
        test_multiple_checkpoints(configs)



if __name__ == "__main__":
    main()
