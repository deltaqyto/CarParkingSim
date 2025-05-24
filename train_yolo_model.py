import os
import sys
import argparse
import shutil
import yaml
import glob
import random
from pathlib import Path

def setup_directories(base_path):
    """Create the necessary directory structure for YOLO training"""
    directories = [
        'data/train/images',
        'data/train/labels', 
        'data/validation/images',
        'data/validation/labels',
        'data/test/images',
        'data/test/labels'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def find_dataset_files(source_path):
    """Find all image and label files in the source directory"""
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
    label_extensions = ['.txt']
    
    images = []
    labels = []
    
    # Search recursively for images and labels
    for root, dirs, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            
            if ext in image_extensions:
                images.append(file_path)
            elif ext in label_extensions and file != 'classes.txt':
                labels.append(file_path)
    
    return images, labels

def match_images_labels(images, labels):
    """Match image files with their corresponding label files"""
    matched_pairs = []
    
    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Find corresponding label file
        label_path = None
        for lbl_path in labels:
            lbl_name = os.path.splitext(os.path.basename(lbl_path))[0]
            if img_name == lbl_name:
                label_path = lbl_path
                break
        
        if label_path:
            matched_pairs.append((img_path, label_path))
        else:
            print(f"Warning: No label found for image {img_path}")
    
    return matched_pairs

def split_dataset(matched_pairs, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """Split dataset into train, validation, and test sets"""
    random.shuffle(matched_pairs)
    
    total = len(matched_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = matched_pairs[:train_end]
    val_pairs = matched_pairs[train_end:val_end]
    test_pairs = matched_pairs[val_end:]
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_pairs)} samples")
    print(f"  Validation: {len(val_pairs)} samples") 
    print(f"  Test: {len(test_pairs)} samples")
    
    return train_pairs, val_pairs, test_pairs

def copy_files(pairs, dest_img_dir, dest_label_dir):
    """Copy image and label files to destination directories"""
    for img_path, label_path in pairs:
        # Copy image
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(dest_img_dir, img_name))
        
        # Copy label
        label_name = os.path.basename(label_path)
        shutil.copy2(label_path, os.path.join(dest_label_dir, label_name))

def find_classes_file(source_path):
    """Find the classes.txt file in the source directory"""
    for root, dirs, files in os.walk(source_path):
        if 'classes.txt' in files:
            return os.path.join(root, 'classes.txt')
    return None

def read_classes(classes_file_path):
    """Read class names from classes.txt file"""
    if not classes_file_path or not os.path.exists(classes_file_path):
        print("Warning: classes.txt not found. Using default class names.")
        return ['object']
    
    with open(classes_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    return classes

def create_data_yaml(classes, output_path):
    """Create the data.yaml configuration file for YOLO training"""
    data = {
        'path': os.path.abspath('data'),
        'train': 'train/images',
        'val': 'validation/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Created data.yaml with {len(classes)} classes: {classes}")

def train_model(data_yaml_path, model_size='yolo11s.pt', epochs=60, imgsz=640):
    """Train the YOLO model"""
    try:
        # Install ultralytics if not already installed
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
    
    # Run training command
    train_command = f"yolo detect train data={data_yaml_path} model={model_size} epochs={epochs} imgsz={imgsz}"
    print(f"Starting training with command: {train_command}")
    
    result = os.system(train_command)
    
    if result == 0:
        print("Training completed successfully!")
        print("Model saved to: runs/detect/train/weights/best.pt")
    else:
        print("Training failed. Please check the error messages above.")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model from CarParkingSim dataset')
    parser.add_argument('--source', required=True, help='Path to CarParkingSim->YOLO folder containing the dataset')
    parser.add_argument('--model', default='yolo11s.pt', help='YOLO model size (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio for training set')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio for validation set')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Ratio for test set')
    
    args = parser.parse_args()
    
    # Validate source path
    if not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        sys.exit(1)
    
    print(f"Processing dataset from: {args.source}")
    
    # Step 1: Setup directory structure
    print("\n1. Setting up directory structure...")
    setup_directories('.')
    
    # Step 2: Find all images and labels
    print("\n2. Finding dataset files...")
    images, labels = find_dataset_files(args.source)
    print(f"Found {len(images)} images and {len(labels)} label files")
    
    if len(images) == 0:
        print("Error: No images found in the source directory.")
        sys.exit(1)
    
    # Step 3: Match images with labels
    print("\n3. Matching images with labels...")
    matched_pairs = match_images_labels(images, labels)
    print(f"Successfully matched {len(matched_pairs)} image-label pairs")
    
    if len(matched_pairs) == 0:
        print("Error: No matching image-label pairs found.")
        sys.exit(1)
    
    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Step 5: Copy files to appropriate directories
    print("\n5. Copying files...")
    copy_files(train_pairs, 'data/train/images', 'data/train/labels')
    copy_files(val_pairs, 'data/validation/images', 'data/validation/labels')
    copy_files(test_pairs, 'data/test/images', 'data/test/labels')
    
    # Step 6: Find and read classes
    print("\n6. Processing class information...")
    classes_file = find_classes_file(args.source)
    classes = read_classes(classes_file)
    
    # Step 7: Create data.yaml
    print("\n7. Creating data.yaml configuration...")
    create_data_yaml(classes, 'data.yaml')
    
    # Step 8: Train the model
    print("\n8. Starting model training...")
    train_model('data.yaml', args.model, args.epochs, args.imgsz)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Your trained model is saved at: runs/detect/train/weights/best.pt")
    print("Training results and metrics are saved in: runs/detect/train/")
    print("\nTo test your model, you can use the yolo_detect.py script:")
    print(f"python yolo_detect.py --model runs/detect/train/weights/best.pt --source path/to/test/image.jpg")

if __name__ == "__main__":
    main()