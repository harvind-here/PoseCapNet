import os
from ultralytics import YOLO
import torch
import random
import yaml

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a pretrained YOLOv5 model
    model = YOLO('yolov5s.pt')

    # Hardcode the dataset path
    train_data = "C:/Users/harvi/object_detection/datasets/coco/images/train2017"
    
    # Use the path in your training function
    model.train(data=train_data, epochs=100, imgsz=640)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to coco.yaml
    coco_yaml_path = os.path.join(script_dir, 'coco.yaml')

    # Load the COCO dataset configuration
    with open(coco_yaml_path, 'r') as file:
        coco_config = yaml.safe_load(file)

    # Get the list of training images
    train_images = coco_config['train']

    # Randomly select 5000 images
    selected_images = random.sample(train_images, 5000)

    # Create a new temporary YAML file with the reduced dataset
    temp_config = coco_config.copy()
    temp_config['train'] = selected_images

    with open('coco_reduced.yaml', 'w') as file:
        yaml.dump(temp_config, file)

    # Fine-tune the model on the reduced COCO dataset
    results = model.train(data='coco_reduced.yaml', epochs=10, imgsz=320, device=device, batch=8, workers=4)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()