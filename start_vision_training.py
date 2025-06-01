from AI.vision_train_utils import do_vision_training


def start_vision_training():
    vision_params = {
        'rl_model_name': 'model name',
        'vision_buffer_size': 2000,
        'vision_collection_threads': 2,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'save_freq': 500,
        'show_random': 1
        
    }

    trainer = do_vision_training([yourenv], vision_params, "VIS_003c")
    return trainer


if __name__ == "__main__":
    start_vision_training()
