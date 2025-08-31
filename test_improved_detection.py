#!/usr/bin/env python3
"""
Test script for improved cell detection
"""

import os
import time
from models import detect_all_cells_comprehensive, load_yolo_model

def test_improved_detection():
    """Test the improved detection system"""
    
    # Find a test image
    test_image = None
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and any(keyword in file.lower() for keyword in ['blood', 'cell', 'smear']):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        print("❌ No test image found")
        return
    
    print(f"🧪 Testing improved detection with: {test_image}")
    print("=" * 60)
    
    # Load model
    print("📥 Loading YOLO model...")
    model = load_yolo_model()
    
    # Test detection
    print("🔍 Running improved detection...")
    start_time = time.time()
    
    results = detect_all_cells_comprehensive(model, test_image, confidence_threshold=0.005)
    
    detection_time = time.time() - start_time
    
    if results:
        stats = results['stats']
        print(f"\n🎯 DETECTION RESULTS:")
        print(f"   Total Cells: {stats['total_cells_detected']}")
        print(f"   RBC: {stats['RBC_count']}")
        print(f"   WBC: {stats['WBC_count']}")
        print(f"   Platelets: {stats['Platelet_count']}")
        print(f"   Detection Time: {detection_time:.2f} seconds")
        print(f"   Detection Density: {stats['detection_density']:.6f} cells/pixel")
        print(f"   Overall Confidence: {stats['confidence_scores']['Overall']:.2%}")
        
        # Check if we're detecting a good number of cells
        total_cells = stats['total_cells_detected']
        if total_cells > 500:
            print(f"\n✅ EXCELLENT: Detected {total_cells} cells - high sensitivity achieved!")
        elif total_cells > 300:
            print(f"\n✅ GOOD: Detected {total_cells} cells - good sensitivity")
        elif total_cells > 100:
            print(f"\n⚠️  MODERATE: Detected {total_cells} cells - could be improved")
        else:
            print(f"\n❌ LOW: Only detected {total_cells} cells - needs optimization")
            
    else:
        print("❌ Detection failed")

if __name__ == "__main__":
    test_improved_detection()