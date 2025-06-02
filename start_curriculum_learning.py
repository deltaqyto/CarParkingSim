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

        from modules.parking_modules import ParkingSchedule2

        do_curriculum_learning(ParkingSchedule2(render=False), override_file_name="APYX")


    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        plt.close('all')
