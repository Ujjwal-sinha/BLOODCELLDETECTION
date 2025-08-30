#!/usr/bin/env python3
"""
Debug script to test blood cell detection
"""

import os
import sys
from PIL import Image
import tempfile

# Import our detection functions
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def debug_detection():
    """Debug the blood cell detection"""
    
    print("🔬 Debugging Blood Cell Detection")
    print("=" * 50)
    
    # Use the first available image
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return False
    
    print(f"📷 Using sample image: {sample_image}")
    
    # Load YOLO model
    print("\n📥 Loading YOLO model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("❌ Failed to load YOLO model - this is the issue!")
        print("🔄 The model is falling back to computer vision detection")
        
        # Test fallback detection directly
        print("\n🔍 Testing fallback detection...")
        from models import create_fallback_detection
        
        fallback_results = create_fallback_detection(sample_image)
        
        if fallback_results:
            stats = fallback_results['stats']
            print(f"✅ Fallback detection found {stats['total_cells_detected']} cells")
            print(f"   RBC: {stats['RBC_count']}")
            print(f"   WBC: {stats['WBC_count']}")
            print(f"   Platelets: {stats['Platelet_count']}")
            
            # Create visualization
            viz_path = visualize_all_cells(sample_image, fallback_results, output_path="debug_fallback_result.png")
            if viz_path:
                print(f"✅ Visualization saved: {viz_path}")
        else:
            print("❌ Even fallback detection failed")
        
        return False
    
    print("✅ YOLO model loaded successfully")
    
    # Test detection with very low threshold
    print("\n🔍 Running detection with very low threshold...")
    detection_results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if not detection_results:
        print("❌ Detection failed completely")
        return False
    
    # Display results
    stats = detection_results['stats']
    print(f"\n📊 Detection Results:")
    print(f"   Total Cells: {stats['total_cells_detected']}")
    print(f"   RBC: {stats['RBC_count']}")
    print(f"   WBC: {stats['WBC_count']}")
    print(f"   Platelets: {stats['Platelet_count']}")
    print(f"   Method: {detection_results.get('detection_method', 'Unknown')}")
    
    # Create visualization
    viz_path = visualize_all_cells(sample_image, detection_results, output_path="debug_detection_result.png")
    if viz_path:
        print(f"✅ Visualization saved: {viz_path}")
    
    return True

if __name__ == "__main__":
    debug_detection()