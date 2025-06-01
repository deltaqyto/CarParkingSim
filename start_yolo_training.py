from AI.YOLO.train_yolo_model import prepare_and_train_yolo

if __name__ == "__main__":
    model_name = "DET_001"  # Set to a preferred name

    # Run the training with default parameters
    prepare_and_train_yolo(
        model_size='yolo11s.pt',
        epochs=30,
        imgsz=640,
        model_name=model_name
    )
