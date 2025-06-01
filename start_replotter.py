from scripts.replot_training import test_multiple_checkpoints
from math import pi
from Simulation.environments import get_basic_env

def param_converter(**kwargs):
    return kwargs

def main():
    env = get_basic_env
    params = {"goal_size":2, "angle_tolerance":pi / 4}
    configs=[[env, params, 'model name', 100, True]]  # env, parameters for env, model, episodes per checkpoint, test all checkpoints?. Add multiple for more model testing
    test_multiple_checkpoints(configs)



if __name__ == "__main__":
    main()
