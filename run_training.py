from ultralytics import YOLO
import os

def train_yolo_model():
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("Error: data.yaml not found. Please run the dataset preparation first.")
        return
    
    # Load a model
    model = YOLO('yolo11s.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    print("Starting YOLO training...")
    results = model.train(
        data='data.yaml',      # path to dataset YAML
        epochs=60,             # number of training epochs
        imgsz=640,             # training image size
        plots=True,            # save plots
        save=True,             # save train checkpoints
        project='runs/detect', # project name
        name='train'           # experiment name
    )
    
    print("Training completed!")
    print("Model saved to: runs/detect/train/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_yolo_model()