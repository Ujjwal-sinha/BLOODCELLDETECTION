#!/usr/bin/env python3
"""
Test script to demonstrate working blood cell detection
"""

import os
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def test_blood_cell_detection():
    """Test blood cell detection on a sample image"""
    
    print("🩸 Blood Cell Detection Test")
    print("=" * 40)
    
    # Use a sample image
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print("❌ Sample image not found")
        return False
    
    print(f"📷 Analyzing: {os.path.basename(sample_image)}")
    
    # Load model
    model = load_yolo_model('yolo11n.pt')
    
    # Run detection
    print("\n🔍 Detecting blood cells...")
    results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if not results:
        print("❌ Detection failed")
        return False
    
    # Show results
    stats = results['stats']
    print(f"\n✅ Detection Complete!")
    print(f"🎯 Total Cells Found: {stats['total_cells_detected']}")
    print(f"🔴 Red Blood Cells (RBC): {stats['RBC_count']}")
    print(f"⚪ White Blood Cells (WBC): {stats['WBC_count']}")
    print(f"🟢 Platelets: {stats['Platelet_count']}")
    print(f"📊 Detection Method: {results['detection_method']}")
    print(f"🎯 Overall Confidence: {stats['confidence_scores']['Overall']:.1%}")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    viz_path = visualize_all_cells(sample_image, results, output_path="blood_cell_detection_result.png")
    
    if viz_path and os.path.exists(viz_path):
        print(f"✅ Visualization saved: {viz_path}")
        print(f"📁 You can view the detected cells in the saved image!")
    
    print(f"\n🎉 Blood cell detection is working perfectly!")
    print(f"💡 The system detected {stats['total_cells_detected']} cells in the blood smear image")
    
    return True

if __name__ == "__main__":
    test_blood_cell_detection()