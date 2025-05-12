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

        # Import the training schedule here, not at the top (avoids circular import)
        from Simulation.training_schedule import BasicTrainingSchedule

        do_curriculum_learning(BasicTrainingSchedule())  # Start training with the 'Basic Training Schedule'
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        plt.close('all')
