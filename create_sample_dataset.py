#!/usr/bin/env python3
"""
Sample Dataset Creator for BloodCellAI
Creates a sample YOLO format dataset structure for blood cell detection
"""

import os
import yaml

def create_sample_dataset():
    """Create sample dataset structure for blood cell detection"""
    
    # Define dataset structure
    dataset_dir = "dataset"
    splits = ["train", "valid", "test"]
    subfolders = ["images", "labels"]
    
    # Create directory structure
    for split in splits:
        for subfolder in subfolders:
            path = os.path.join(dataset_dir, split, subfolder)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    # Create data.yaml configuration file
    data_yaml = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 3,  # number of classes
        'names': {
            0: 'RBC',
            1: 'WBC', 
            2: 'Platelets'
        }
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created configuration file: {yaml_path}")
    
    # Create README with instructions
    readme_content = """# BloodCellAI Dataset

This directory contains the dataset structure for blood cell detection using YOLO format.

## Directory Structure
```
dataset/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/         # Training images (.jpg, .png)
│   └── labels/         # Training labels (.txt)
├── valid/
│   ├── images/         # Validation images
│   └── labels/         # Validation labels
└── test/
    ├── images/         # Test images
    └── labels/         # Test labels
```

## Label Format
Each label file (.txt) should contain one line per object in YOLO format:
```
class_id x_center y_center width height
```

Where:
- class_id: 0=RBC, 1=WBC, 2=Platelets
- x_center, y_center: center coordinates (normalized 0-1)
- width, height: bounding box dimensions (normalized 0-1)

## Example Label File
```
0 0.5 0.3 0.1 0.1    # RBC at center (0.5, 0.3) with size 0.1x0.1
1 0.2 0.7 0.15 0.15  # WBC at (0.2, 0.7) with size 0.15x0.15
2 0.8 0.4 0.05 0.05  # Platelet at (0.8, 0.4) with size 0.05x0.05
```

## Getting Started
1. Add your blood smear images to the respective images/ folders
2. Create corresponding label files in the labels/ folders
3. Run the BloodCellAI application to train and analyze

## Blood Cell Classes
- **RBC (Red Blood Cells)**: Oxygen-carrying cells, typically round and biconcave
- **WBC (White Blood Cells)**: Immune system cells, larger with visible nuclei
- **Platelets**: Small cell fragments responsible for blood clotting
"""
    
    readme_path = os.path.join(dataset_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README file: {readme_path}")
    
    print("\n✅ Sample dataset structure created successfully!")
    print(f"📁 Dataset directory: {dataset_dir}")
    print("\n📋 Next steps:")
    print("1. Add your blood smear images to the images/ folders")
    print("2. Create corresponding YOLO format label files")
    print("3. Run the BloodCellAI application")

if __name__ == "__main__":
    create_sample_dataset()