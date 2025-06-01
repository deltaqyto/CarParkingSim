from AI.YOLO.train_yolo_model import prepare_and_train_yolo

if __name__ == "__main__":
    import os

    # Use a relative path that should work on any machine
    source_path = os.path.join("YOLO")

    # Create paths for working directory and model output
    working_dir = "temp_data"
    model_dir = "models/YOLO"

    # Display information for the user
    print(f"Looking for YOLO dataset in: {os.path.abspath(source_path)}")

    # Run the training with default parameters
    prepare_and_train_yolo(
        source_path=source_path,
        model_size='yolo11s.pt',
        epochs=60,
        imgsz=640,
        working_dir=working_dir,
        model_dir=model_dir
    )
