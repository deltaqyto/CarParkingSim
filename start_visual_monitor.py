import multiprocessing as mp
import matplotlib

matplotlib.use('Agg')
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from AI.visual_monitor import VisualMonitor

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)

        from Simulation.training_schedule import YOLOParkingSchedule
        schedule = YOLOParkingSchedule(render=True)
        model_targets = {    'ZO77_STP_1': 0,}
        monitor = VisualMonitor(    schedule_class=YOLOParkingSchedule,    schedule_kwargs={'render': True},    model_targets=model_targets)
        

        print("Starting visual monitor...")
        print(f"Monitoring models: {list(model_targets.keys())}")
        monitor.run()

    except KeyboardInterrupt:
        print("\nShutting down visual monitor...")
    except Exception as e:
        print(f"Fatal error: {e}")
