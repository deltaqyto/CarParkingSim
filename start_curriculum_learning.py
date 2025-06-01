import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from AI.curriculum_learning import do_curriculum_learning


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)

        from Simulation.training_schedule import YOLOParkingSchedule

        do_curriculum_learning(YOLOParkingSchedule())


    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        plt.close('all')
