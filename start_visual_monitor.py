import multiprocessing as mp
import matplotlib

matplotlib.use('Agg')
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from AI.visual_monitor import VisualMonitor

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)

        from modules.parking_modules import ParkingSchedule2

        model_targets = {
            'Apyx_STP_1': 0,
            'Apyx_STP_2': 1,
            'Apyx_STP_3': 2,
            'Apyx_STP_4': 3,
            'Apyx_STP_5': 4,
            'Apyx_STP_6': 5,
            'Apyx_STP_7': 6,
            'Apyx_STP_8': 7,
        }

        # Start the visual monitor
        monitor = VisualMonitor(
            schedule_class=ParkingSchedule2,
            schedule_kwargs={'render': True},#, 'vision_model':"VIS_006b"},
            model_targets=model_targets
        )

        print("Starting visual monitor...")
        print(f"Monitoring models: {list(model_targets.keys())}")
        monitor.run()

    except KeyboardInterrupt:
        print("\nShutting down visual monitor...")
    except Exception as e:
        print(f"Fatal error: {e}")
