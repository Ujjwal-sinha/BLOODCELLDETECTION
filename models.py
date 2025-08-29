"""
Blood Cell Detection Models - BloodCellAI
YOLO11n model for blood cell detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def clear_mps_cache():
    """Clear MPS cache to prevent memory issues"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

class BloodCellDataset(Dataset):
    """
    Custom Dataset for blood cell images and YOLO annotations
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.classes = ['Platelets', 'RBC', 'WBC']
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        image = Image.open(img_path).convert('RGB')
        
        # Load YOLO annotations
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))
        
        if self.transform:
            image = self.transform(image)
        
        return image, {'boxes': boxes, 'labels': labels}

def load_yolo_model(model_path, num_classes=3):
    """
    Load YOLO11n model for blood cell detection
    """
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"Loaded YOLO11n model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def train_model(model, data_yaml, num_epochs=10, batch_size=16):
    """
    Train the YOLO11n model
    """
    try:
        model.train(
            data=data_yaml,
            epochs=num_epochs,
            batch=batch_size,
            imgsz=640,
            device=device
        )
        
        model.save('best_blood_cell_model.pt')
        
        return {'status': 'Training completed'}
    except Exception as e:
        print(f"Error training model: {e}")
        return {'status': f'Error: {str(e)}'}

def evaluate_model(model, data_loader, classes, device='cpu'):
    """
    Evaluate the YOLO model
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            results = model.predict(images, conf=0.5)
            
            for i, result in enumerate(results):
                pred_boxes = result.boxes.xyxy.cpu().numpy()
                pred_labels = result.boxes.cls.cpu().numpy()
                pred_scores = result.boxes.conf.cpu().numpy()
                
                all_predictions.extend([(box, label, score) for box, label, score in zip(pred_boxes, pred_labels, pred_scores)])
                all_labels.extend([(box, label) for box, label in zip(targets['boxes'][i], targets['labels'][i])])
    
    if not all_predictions or not all_labels:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'predictions': [], 'true_labels': []}
    
    precision = precision_score([int(x[1]) for x in all_predictions], [int(x[1]) for x in all_labels], average='macro')
    recall = recall_score([int(x[1]) for x in all_predictions], [int(x[1]) for x in all_labels], average='macro')
    f1 = f1_score([int(x[1]) for x in all_predictions], [int(x[1]) for x in all_labels], average='macro')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'true_labels': all_labels
    }

def predict_blood_cells(model, image, classes, confidence_threshold=0.5):
    """
    Predict blood cells in a single image
    """
    try:
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            results = model.predict(image_tensor, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                detections.append({
                    'box': box,
                    'class': classes[int(label)],
                    'confidence': float(score)
                })
        
        return {'detections': detections}
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {'detections': []}

def plot_training_history(history):
    """
    Plot training history (placeholder for YOLO metrics)
    """
    pass

def plot_confusion_matrix(conf_matrix, classes, save_path='blood_cell_confusion_matrix.png'):
    """
    Plot confusion matrix for blood cell detection
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Blood Cell Detection Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()