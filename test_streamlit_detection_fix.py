#!/usr/bin/env python3
"""
Test the detection fix for Streamlit app
"""

import os
import tempfile
from models import detect_all_cells_comprehensive, load_yolo_model
from PIL import Image

def test_detection_fix():
    """Test that detection works like in the reference image"""
    
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
    
    print(f"🧪 Testing detection fix with: {test_image}")
    print("=" * 60)
    
    # Load model (same as Streamlit app)
    print("📥 Loading YOLO model...")
    yolo_model = load_yolo_model()
    
    # Create temp file (same as Streamlit app)
    image = Image.open(test_image)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        
        # Run detection (same as Streamlit app)
        print("🔍 Running detection (same as Streamlit)...")
        detection_results = detect_all_cells_comprehensive(yolo_model, tmp_file.name, confidence_threshold=0.005)
        
        # Clean up
        os.unlink(tmp_file.name)
    
    if detection_results:
        stats = detection_results['stats']
        print(f"\n🎯 DETECTION RESULTS:")
        print(f"   Total Cells: {stats['total_cells_detected']}")
        print(f"   RBC: {stats['RBC_count']}")
        print(f"   WBC: {stats['WBC_count']}")
        print(f"   Platelets: {stats['Platelet_count']}")
        print(f"   Detection Method: {detection_results.get('detection_method', 'Unknown')}")
        
        # Check if we're getting good results like the reference image (605 cells)
        total_cells = stats['total_cells_detected']
        if total_cells > 500:
            print(f"\n✅ EXCELLENT: Detected {total_cells} cells - matches reference quality!")
        elif total_cells > 200:
            print(f"\n✅ GOOD: Detected {total_cells} cells - good detection")
        elif total_cells > 50:
            print(f"\n⚠️  MODERATE: Detected {total_cells} cells - could be better")
        else:
            print(f"\n❌ POOR: Only detected {total_cells} cells - needs fixing")
            
        return total_cells > 200  # Return success if we detect a reasonable number
    else:
        print("❌ Detection failed completely")
        return False

if __name__ == "__main__":
    success = test_detection_fix()
    if success:
        print("\n🎉 Detection fix successful! Should work in Streamlit app.")
    else:
        print("\n❌ Detection fix failed. Needs more work.")