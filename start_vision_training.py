from AI.vision_train_utils import do_vision_training
from modules.parking_modules import ParkingSchedule2


def start_vision_training():
    vision_params = {
        'rl_model_name': 'APYX_STP_7',
        'vision_buffer_size': 2000,
        'vision_collection_threads': 2,
        'batch_size': 64,
        'learning_rate': 5e-5,
        'save_freq': 500,
        'show_random': 0.4,
        'max_batches': 2000,
        
    }

    trainer = do_vision_training(ParkingSchedule2(vision=True).get_nth_environment(6)[0], vision_params, "VIS_006a")
    return trainer


if __name__ == "__main__":
    start_vision_training()
