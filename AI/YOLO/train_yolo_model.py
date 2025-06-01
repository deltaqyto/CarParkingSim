import os
import shutil
import yaml
import random
from ultralytics import YOLO
import string


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
    with open(classes_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    return classes


def create_data_yaml(classes, output_path, data_dir):
    """Create the data.yaml configuration file for YOLO training"""
    data = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'validation/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created data.yaml with {len(classes)} classes: {classes}")


def train_model(data_yaml_path, model_size='yolo11s.pt', epochs=60, imgsz=640, model_dir='models/YOLO', model_name='model'):
    """Train the YOLO model using ultralytics Python API"""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Add a letter suffix if name exists
    base_name = model_name
    full_path = os.path.join(model_dir, model_name)

    if os.path.exists(full_path):
        for letter in string.ascii_lowercase:
            model_name = f"{base_name}{letter}"
            full_path = os.path.join(model_dir, model_name)
            if not os.path.exists(full_path):
                break

    # Load the model
    model = YOLO(model_size)

    # Train the model
    print(f"Starting training with YOLO model: {model_size}")
    print(f"Model will be saved as: {model_name}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project=model_dir,
        name=model_name
    )

    # Get the path to the best weights
    full_model_path = os.path.join(model_dir, model_name)
    best_weights_path = os.path.join(full_model_path, 'weights/best.pt')

    print("Training completed!")
    print(f"Model saved to: {best_weights_path}")

    return best_weights_path


def prepare_and_train_yolo(source_path="YOLO", model_size='yolo11s.pt', epochs=60, imgsz=640,
                           train_ratio=0.8, val_ratio=0.15, test_ratio=0.05,
                           working_dir='temp_data', model_dir='models/YOLO', model_name='unnamed_model'):
    """Main function to prepare dataset and train YOLO model"""
    # Ensure absolute paths
    working_dir = os.path.abspath(working_dir)
    model_dir = os.path.abspath(model_dir)

    # Step 1: Setup directory structure
    print("\n1. Setting up directory structure...")
    setup_directories(working_dir)

    # Step 2: Find all images and labels
    print("\n2. Finding dataset files...")
    images, labels = find_dataset_files(source_path)
    print(f"Found {len(images)} images and {len(labels)} label files")

    # Step 3: Match images with labels
    print("\n3. Matching images with labels...")
    matched_pairs = match_images_labels(images, labels)
    print(f"Successfully matched {len(matched_pairs)} image-label pairs")

    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs, train_ratio, val_ratio, test_ratio
    )

    # Step 5: Copy files to appropriate directories
    print("\n5. Copying files...")
    data_train_img_dir = os.path.join(working_dir, 'data/train/images')
    data_train_lbl_dir = os.path.join(working_dir, 'data/train/labels')
    data_val_img_dir = os.path.join(working_dir, 'data/validation/images')
    data_val_lbl_dir = os.path.join(working_dir, 'data/validation/labels')
    data_test_img_dir = os.path.join(working_dir, 'data/test/images')
    data_test_lbl_dir = os.path.join(working_dir, 'data/test/labels')

    copy_files(train_pairs, data_train_img_dir, data_train_lbl_dir)
    copy_files(val_pairs, data_val_img_dir, data_val_lbl_dir)
    copy_files(test_pairs, data_test_img_dir, data_test_lbl_dir)

    # Step 6: Find and read classes
    print("\n6. Processing class information...")
    classes_file = find_classes_file(source_path)
    classes = read_classes(classes_file)

    # Step 7: Create data.yaml
    print("\n7. Creating data.yaml configuration...")
    yaml_path = os.path.join(working_dir, 'data.yaml')
    create_data_yaml(classes, yaml_path, os.path.join(working_dir, 'data'))

    # Step 8: Train the model
    print("\n8. Starting model training...")
    best_weights_path = train_model(yaml_path, model_size, epochs, imgsz, model_dir, model_name)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    model_folder = os.path.dirname(best_weights_path)
    print(f"Your trained model is saved at: {best_weights_path}")
    print(f"Training results and metrics are saved in: {model_folder}")
    print("\nTo test your model, you can use the yolo_detect.py script:")
    print(f"python yolo_detect.py --model {best_weights_path} --source path/to/test/image.jpg")
