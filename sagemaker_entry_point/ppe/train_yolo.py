# sagemaker_entry_point/train_yolo.py
import os
import sys
import subprocess
import yaml
import shutil
import argparse

def install_dependencies():
    """Installs YOLOv8 (ultralytics) inside the container."""
    print("Installing Ultralytics YOLOv8...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

def update_yaml_paths(data_dir):
    """
    SageMaker mounts S3 data to /opt/ml/input/data/training.
    We need to update data.yaml to point to this absolute path.
    """
    yaml_path = os.path.join(data_dir, 'data.yaml')
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update base path to SageMaker's mount point
    data['path'] = data_dir 
    # Ensure train/val are relative to 'path'
    data['train'] = 'train/images'
    data['val'] = 'valid/images' # Roboflow usually names it 'valid'
    
    # Save the new config
    new_yaml_path = os.path.join(data_dir, 'data_sagemaker.yaml')
    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    print(f"data.yaml updated to point to: {data_dir}")
    return new_yaml_path

def train(args):
    from ultralytics import YOLO

    print(f"Starting YOLOv8 Training on device: {args.device}")
    
    # 1. Prepare Data Configuration
    data_config = update_yaml_paths(args.data_dir)
    
    model = YOLO(args.model_name) 

    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=args.model_dir, # /opt/ml/model
        name='yolo_experiment',  # Subfolder inside model_dir
        exist_ok=True
    )
    
    # 2. Move the best model to the model directory root for SageMaker
    source_path = os.path.join(args.model_dir, 'yolo_experiment', 'weights', 'best.pt')
    target_path = os.path.join(args.model_dir, 'model.pt') # Standardize name
    
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f"Model artifact moved to: {target_path}")
    else:
        print("Could not find best.pt file.")

if __name__ == '__main__':
    install_dependencies()
    
    parser = argparse.ArgumentParser()
    
    # SageMaker Parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--model-name', type=str, default='yolov8n.pt')
    parser.add_argument('--device', type=str, default='0') # '0' for GPU, 'cpu' for CPU
    
    # SageMaker Environment Variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    args = parser.parse_args()
    train(args)