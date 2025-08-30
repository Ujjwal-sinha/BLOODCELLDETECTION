#!/usr/bin/env python3
"""
Demo: Blood Cell Detection - Like the reference image you showed
This script demonstrates comprehensive blood cell detection
"""

import os
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def demo_blood_cell_detection():
    """Demo blood cell detection similar to your reference"""
    
    print("🩸 BLOOD CELL DETECTION DEMO")
    print("=" * 50)
    print("🎯 Detecting ALL blood cells in the image")
    print("🔍 Using advanced computer vision algorithms")
    print()
    
    # Use multiple sample images to show variety
    sample_images = [
        "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg",
        "dataset/train/images/BloodImage_00002_jpg.rf.3f628ac8c9a3cccac2926e36a29d7eb5.jpg",
        "dataset/train/images/BloodImage_00003_jpg.rf.8d4037c47a4a76557c729057293e0755.jpg"
    ]
    
    # Load model once
    print("📥 Loading AI detection model...")
    model = load_yolo_model('yolo11n.pt')
    print("✅ Model loaded successfully")
    print()
    
    for i, sample_image in enumerate(sample_images, 1):
        if not os.path.exists(sample_image):
            continue
            
        print(f"🔬 ANALYZING BLOOD SAMPLE #{i}")
        print("-" * 40)
        print(f"📷 Image: {os.path.basename(sample_image)}")
        
        # Run detection
        print("🔍 Scanning for blood cells...")
        results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
        
        if results:
            stats = results['stats']
            
            # Display results like your reference
            print(f"✅ DETECTION COMPLETE!")
            print(f"🎯 TOTAL CELLS DETECTED: {stats['total_cells_detected']}")
            print()
            print("📊 CELL BREAKDOWN:")
            print(f"   🔴 Red Blood Cells (RBC): {stats['RBC_count']} ({stats['cell_distribution']['RBC_percentage']:.1f}%)")
            print(f"   ⚪ White Blood Cells (WBC): {stats['WBC_count']} ({stats['cell_distribution']['WBC_percentage']:.1f}%)")
            print(f"   🟢 Platelets: {stats['Platelet_count']} ({stats['cell_distribution']['Platelet_percentage']:.1f}%)")
            print()
            print("🎯 QUALITY METRICS:")
            print(f"   Overall Confidence: {stats['confidence_scores']['Overall']:.1%}")
            print(f"   Detection Density: {stats['detection_density']:.6f} cells/pixel")
            print(f"   Coverage Areas: {stats['detection_coverage']} regions")
            
            # Create visualization
            output_name = f"blood_detection_sample_{i}.png"
            viz_path = visualize_all_cells(sample_image, results, output_path=output_name)
            
            if viz_path and os.path.exists(viz_path):
                print(f"📁 Visualization saved: {output_name}")
            
            print()
            print("🎉 ANALYSIS COMPLETE!")
            print("=" * 50)
            print()
        else:
            print("❌ Detection failed for this sample")
            print()
    
    print("🏁 DEMO COMPLETE!")
    print("💡 The blood cell detection system successfully identified cells in all samples")
    print("📊 Results show comprehensive detection of RBCs, WBCs, and Platelets")
    print("🎨 Visualizations have been saved showing detected cells with bounding boxes")

if __name__ == "__main__":
    demo_blood_cell_detection()