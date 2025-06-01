import os
import random
import string
import torch
from time import time

# Import shared functionality
from AI.train_utils import setup_model_training, train_model


def do_curriculum_learning(curriculum, override_file_name=None, search_path="models", continue_from=None):
    # Handle continue_from logic
    if continue_from:
        base_train_id, start_step = parse_continue_from(continue_from)
        base_train_id = base_train_id if override_file_name is None else override_file_name
        previous_train_id = continue_from
    else:
        base_train_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) if override_file_name is None else override_file_name
        start_step = 1
        previous_train_id = None

    start_time = time()

    print("\n======= TD3 Car Curriculum Trainer=======\n")

    for lesson_num in range(curriculum.get_num_environments()):
        current_step = start_step + lesson_num
        train_id = generate_unique_train_id(base_train_id, current_step, search_path)
        base_env, params = curriculum.get_next_environment()

        print(f"({current_step}/{start_step + curriculum.get_num_environments() - 1}) Training {train_id} with parameters:")
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

    print(f"Training took {round(time() - start_time)} seconds")


def parse_continue_from(continue_from):
    """Parse continue_from string and return base_train_id and next step number"""
    if '_STP_' in continue_from:
        parts = continue_from.split('_STP_')
        base_code = parts[0]
        step_part = parts[1]

        # Extract step number (ignore letter suffix)
        step_num_str = ''.join(filter(str.isdigit, step_part))
        step_num = int(step_num_str)

        # Return base code and next step
        return base_code, step_num + 1
    else:
        # No step suffix, start from step 1
        return continue_from, 1


def generate_unique_train_id(base_train_id, lesson_num, search_path):
    """Generate unique train_id, adding letter suffix if collision occurs"""
    base_id = f"{base_train_id}_STP_{lesson_num}"

    # Check if base_id is available
    model_dir = os.path.join(search_path, f"td3_{base_id}")
    if not os.path.exists(model_dir):
        return base_id

    # Try letter suffixes
    for letter in 'bcdefghijklmnopqrstuvwxyz':
        candidate_id = f"{base_id}{letter}"
        model_dir = os.path.join(search_path, f"td3_{candidate_id}")
        if not os.path.exists(model_dir):
            return candidate_id

    # If all letters are taken, use timestamp
    import time
    timestamp = str(int(time.time()))[-4:]
    return f"{base_id}_{timestamp}"
