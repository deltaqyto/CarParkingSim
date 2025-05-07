import os
import multiprocessing as mp
import random
import string
import torch
import matplotlib.pyplot as plt

# Import shared functionality
from AI.train_utils import setup_model_training, train_model


def do_curriculum_learning(curriculum):
    search_path = "models"  # Change this if you store your models elsewhere
    override_file_name = None  # Set a model name. Leave as none for auto-generated

    base_train_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) if override_file_name is None else override_file_name
    previous_train_id = None

    print("\n======= TD3 Car Curriculum Trainer=======\n")
    for lesson_num in range(curriculum.get_num_environments()):
        train_id = base_train_id + '_STP_' + str(lesson_num + 1)
        base_env, params = curriculum.get_next_environment()

        print(f"({lesson_num + 1}/{curriculum.get_num_environments()}) Training {train_id} with parameters:")
        print("{")
        for key, value in params.items():
            print(f"    {key}: {value}")
        print("}")

        # Determine if we're loading a previous model
        model_path = None
        if previous_train_id:
            model_path = os.path.join(search_path, f"td3_{previous_train_id}", f"{previous_train_id}_final.zip")

        # Setup model and training environment
        model, env, checkpoint_callback, model_dir, monitor = setup_model_training(
            environment=base_env,
            params=params,
            train_id=train_id,
            model_path=model_path,
            search_path=search_path
        )

        # Train the model
        print(f"Starting training with ID: {train_id}")
        print(f"Models will be saved to: {model_dir}")
        print(f"Using {params['num_envs']} environments on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print(f"Automatic evaluation will run {params['eval_episodes']} episodes per checkpoint")

        final_path, exit_trainer = train_model(
            model=model,
            env=env,
            checkpoint_callback=checkpoint_callback,
            model_dir=model_dir,
            train_id=train_id,
            total_timesteps=params['total_timesteps'],
            monitor=monitor
        )

        previous_train_id = train_id
        if exit_trainer:
            return
