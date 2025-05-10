import os
import random
import string
import torch

# Import shared functionality
from AI.train_utils import find_td3_models, get_best_model, setup_model_training, train_model


def do_single_learning(environment, params):
    search_path = "models"  # Change this if you store your models elsewhere

    print("\n======= TD3 Single Environment Trainer =======\n")

    continue_training = False
    model_to_load = None

    while True:
        existing_model_code = input("Enter model ID to continue training (or leave blank for new training): ").strip().upper()

        if not existing_model_code:
            break

        existing_models = find_td3_models(search_path, existing_model_code)

        if not existing_models:
            print(f"No existing models found with ID {existing_model_code}")
            continue

        print(f"Found {len(existing_models)} checkpoints for model td3_{existing_model_code}")
        continue_training = True

        while True:
            selected_model = input(f"Enter specific checkpoint to load (or leave blank for best available): ").strip()

            if not selected_model:
                model_to_load = get_best_model(existing_models)
                print(f"Using best available model: {os.path.basename(model_to_load)}")
                break

            # Check if user only entered step count (integer)
            if selected_model.isdigit():
                selected_model = f"{existing_model_code}_{selected_model}.zip"
            elif not selected_model.endswith('.zip'):
                selected_model = f"{selected_model}.zip"

            model_path = os.path.join(search_path, f"td3_{existing_model_code}", selected_model)
            if os.path.exists(model_path):
                model_to_load = model_path
                print(f"Selected model: {model_path}")
                break
            else:
                print(f"Model not found: {model_path}. Try again")
        break

    # Generate a random training ID
    train_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    print("\nTraining parameters:")
    print(f"Total timesteps: {params['total_timesteps']}")
    print(f"Environments: {params['num_envs']}")
    print(f"Starting from model: {'None' if not continue_training else os.path.basename(model_to_load)}")
    print(f"New training ID: {train_id}")

    # Setup model and training environment
    model, env, checkpoint_callback, model_dir, monitor = setup_model_training(
        environment=environment,
        params=params,
        train_id=train_id,
        model_path=model_to_load if continue_training else None,
        search_path=search_path
    )

    # Train the model
    print(f"Starting training with ID: {train_id}")
    print(f"Models will be saved to: {model_dir}")
    print(f"Using {params['num_envs']} environments")
    print(f"Automatic evaluation will run {params['eval_episodes']} episodes per checkpoint")

    final_path, _ = train_model(
        model=model,
        env=env,
        checkpoint_callback=checkpoint_callback,
        model_dir=model_dir,
        train_id=train_id,
        total_timesteps=params['total_timesteps'],
        monitor=monitor
    )

    return final_path
