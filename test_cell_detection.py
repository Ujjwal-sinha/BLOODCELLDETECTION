#!/usr/bin/env python3
"""
Test script for comprehensive blood cell detection
"""

import os
import sys
from PIL import Image
import tempfile

# Import our detection functions
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def test_comprehensive_detection():
    """Test the comprehensive cell detection function"""
    
    print("🔬 Testing Comprehensive Blood Cell Detection")
    print("=" * 50)
    
    # Load YOLO model
    print("📥 Loading YOLO model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("❌ Failed to load YOLO model")
        return False
    
    print("✅ YOLO model loaded successfully")
    
    # Check if we have sample images
    sample_dirs = ['dataset/train/images', 'dataset/valid/images', 'dataset/test/images']
    sample_image = None
    
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                sample_image = os.path.join(sample_dir, images[0])
                break
    
    if not sample_image:
        print("❌ No sample images found in dataset directories")
        return False
    
    print(f"📷 Using sample image: {sample_image}")
    
    # Run comprehensive detection
    print("\n🔍 Running comprehensive cell detection...")
    detection_results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.1)
    
    if not detection_results:
        print("❌ Detection failed")
        return False
    
    # Display results
    stats = detection_results['stats']
    print("\n📊 Detection Results:")
    print(f"   Total Cells Detected: {stats['total_cells_detected']}")
    print(f"   RBC Count: {stats['RBC_count']}")
    print(f"   WBC Count: {stats['WBC_count']}")
    print(f"   Platelet Count: {stats['Platelet_count']}")
    print(f"   Overall Confidence: {stats['confidence_scores']['Overall']:.2%}")
    print(f"   Detection Density: {stats['detection_density']:.6f} cells/pixel")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    viz_path = visualize_all_cells(sample_image, detection_results, output_path="test_detection_result.png")
    
    if viz_path and os.path.exists(viz_path):
        print(f"✅ Visualization saved to: {viz_path}")
    else:
        print("⚠️ Could not create visualization")
    
    print("\n🎉 Comprehensive detection test completed!")
    return True

if __name__ == "__main__":
    success = test_comprehensive_detection()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)