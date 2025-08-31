import torch
from ultralytics import YOLO
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import pandas as pd
from google.colab import drive

def mount_drive():
    """Mount Google Drive to access dataset"""
    drive.mount('/content/drive')

def setup_training():
    """Setup the training environment and return the data config"""
    # Load the dataset configuration
    with open('dataset.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    return data_config

def plot_training_metrics(results):
    """
    Plot training metrics including loss curves and performance metrics
    
    Args:
        results: Training results from YOLO model
    """
    metrics = pd.DataFrame(results.results_dict)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training loss
    axes[0, 0].plot(metrics['epoch'], metrics['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(metrics['epoch'], metrics['train/cls_loss'], label='Class Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot validation metrics
    axes[0, 1].plot(metrics['epoch'], metrics['metrics/precision'], label='Precision')
    axes[0, 1].plot(metrics['epoch'], metrics['metrics/recall'], label='Recall')
    axes[0, 1].plot(metrics['epoch'], metrics['metrics/mAP50'], label='mAP@0.5')
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    
    # Plot learning rate
    axes[1, 0].plot(metrics['epoch'], metrics['lr/pg0'], label='Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    
    # Plot mAP by class
    if 'metrics/mAP50-per-class' in metrics.columns:
        axes[1, 1].bar(['RBC', 'WBC', 'Platelets'], 
                      metrics['metrics/mAP50-per-class'].iloc[-1])
        axes[1, 1].set_title('mAP50 by Class (Final)')
        axes[1, 1].set_ylabel('mAP@0.5')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def analyze_model_performance(model, val_loader):
    """
    Analyze model performance on validation set
    
    Args:
        model: Trained YOLO model
        val_loader: Validation data loader
    """
    results = model.val()
    
    # Create detailed performance report
    performance_report = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.precision,
        'Recall': results.box.recall,
        'F1-Score': 2 * (results.box.precision * results.box.recall) / 
                    (results.box.precision + results.box.recall)
    }
    
    # Per-class metrics
    class_names = ['RBC', 'WBC', 'Platelets']
    class_metrics = pd.DataFrame({
        'Precision': results.box.class_precision,
        'Recall': results.box.class_recall,
        'mAP50': results.box.class_map50
    }, index=class_names)
    
    return performance_report, class_metrics

def train_model(epochs=100, batch_size=16, img_size=640):
    """
    Train the YOLO model for blood cell detection
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
    """
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    
    # Training arguments
    args = {
        'data': 'dataset.yaml',  # Path to data configuration file
        'epochs': epochs,        # Number of epochs
        'batch': batch_size,     # Batch size
        'imgsz': img_size,      # Image size
        'patience': 50,         # Early stopping patience
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device to use
        'workers': 8,           # Number of worker threads
        'project': 'blood_cell_detection',  # Project name
        'name': 'training_run',  # Run name
        'exist_ok': True,       # Overwrite existing run
        'pretrained': True,     # Use pretrained model
        'optimizer': 'Adam',    # Optimizer
        'lr0': 0.001,          # Initial learning rate
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 7.5,            # Box loss gain
        'cls': 0.5,            # Cls loss gain
        'dfl': 1.5,            # DFL loss gain
        'save': True,          # Save train checkpoints
        'save_period': -1,     # Save checkpoint every x epochs (-1 for last epoch only)
        'plots': True,         # Save training plots
        'verbose': True,       # Verbose output
    }
    
    # Train the model
    results = model.train(**args)
    
    # Plot training metrics
    plot_training_metrics(results)
    
    # Analyze model performance
    performance_report, class_metrics = analyze_model_performance(model, None)
    
    return results, performance_report, class_metrics

def visualize_predictions(model, test_images, save_dir='prediction_analysis'):
    """
    Visualize model predictions on test images
    
    Args:
        model: Trained YOLO model
        test_images: List of test image paths
        save_dir: Directory to save visualization results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for img_path in test_images[:5]:  # Visualize first 5 test images
        results = model.predict(img_path)[0]
        
        # Plot the image with predictions
        fig, ax = plt.subplots(figsize=(12, 8))
        img = plt.imread(img_path)
        plt.imshow(img)
        
        # Draw bounding boxes and labels
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color=['r', 'g', 'b'][cls])
            ax.add_patch(rect)
            plt.text(x1, y1, f'{["RBC", "WBC", "Platelets"][cls]}: {conf:.2f}', 
                    color='white', backgroundcolor='black')
        
        plt.axis('off')
        plt.savefig(f'{save_dir}/prediction_{os.path.basename(img_path)}')
        plt.close()

def generate_analysis_report(performance_report, class_metrics):
    """
    Generate a detailed analysis report
    
    Args:
        performance_report: Dictionary containing overall performance metrics
        class_metrics: DataFrame containing per-class metrics
    """
    report = "Blood Cell Detection - Model Analysis Report\n"
    report += "=" * 50 + "\n\n"
    
    # Overall Performance
    report += "Overall Performance Metrics:\n"
    report += "-" * 30 + "\n"
    for metric, value in performance_report.items():
        report += f"{metric}: {value:.4f}\n"
    report += "\n"
    
    # Per-class Performance
    report += "Per-class Performance Metrics:\n"
    report += "-" * 30 + "\n"
    report += class_metrics.to_string()
    report += "\n\n"
    
    # Save report
    with open('analysis_report.txt', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main function to run the training pipeline"""
    # Mount Google Drive
    mount_drive()
    
    # Setup training
    data_config = setup_training()
    print("Dataset configuration loaded:", data_config)
    
    # Train model
    print("Starting model training...")
    results, performance_report, class_metrics = train_model(epochs=100)
    print("Training completed!")
    
    # Generate and save analysis
    print("\nGenerating analysis report...")
    report = generate_analysis_report(performance_report, class_metrics)
    print("Analysis report generated and saved as 'analysis_report.txt'")
    
    # Visualize predictions on test set
    print("\nGenerating prediction visualizations...")
    test_images = [f for f in os.listdir(data_config['test']) 
                  if f.endswith(('.jpg', '.png'))]
    test_images = [os.path.join(data_config['test'], f) for f in test_images]
    visualize_predictions(results.model, test_images)
    print("Prediction visualizations saved in 'prediction_analysis' directory")
    
    print("\nModel and analysis files saved in 'blood_cell_detection/training_run'")

if __name__ == "__main__":
    main()
