from AI.YOLO.train_yolo_model import prepare_and_train_yolo

if __name__ == "__main__":
    model_name = "DET_001"  # Set to a preferred name

    # Run the training with default parameters
    prepare_and_train_yolo(
        model_size='yolo11s.pt',  # load a pretrained model (recommended for training)
        epochs=30,                # number of training epochs
        imgsz=640,                # training image size
        model_name=model_name
    )
