#!/usr/bin/env python3
"""
Test the detection functionality that would be used in Streamlit
"""

import os
import tempfile
from PIL import Image
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def test_streamlit_workflow():
    """Test the same workflow that Streamlit would use"""
    
    print("🔬 Testing Streamlit Detection Workflow")
    print("=" * 50)
    
    # Use a sample image
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print("❌ Sample image not found")
        return False
    
    # Load image like Streamlit would
    print("📷 Loading image...")
    image = Image.open(sample_image)
    print(f"✅ Image loaded: {image.size}")
    
    # Save to temp file like Streamlit does
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    print(f"💾 Saved to temp file: {tmp_path}")
    
    # Load model
    print("\n📥 Loading YOLO model...")
    yolo_model = load_yolo_model('yolo11n.pt')
    
    if yolo_model is None:
        print("❌ Failed to load YOLO model")
        return False
    
    print("✅ YOLO model loaded")
    
    # Run detection like Streamlit does
    print("\n🔍 Running comprehensive blood cell detection...")
    try:
        detection_results = detect_all_cells_comprehensive(yolo_model, tmp_path, confidence_threshold=0.01)
        
        if detection_results:
            stats = detection_results['stats']
            
            print("✅ Detection successful!")
            print(f"🎯 Total cells: {stats['total_cells_detected']}")
            print(f"🔴 RBC: {stats['RBC_count']}")
            print(f"⚪ WBC: {stats['WBC_count']}")
            print(f"🟢 Platelets: {stats['Platelet_count']}")
            
            # Test visualization
            print("\n🎨 Creating visualization...")
            viz_path = visualize_all_cells(tmp_path, detection_results, output_path="streamlit_test_result.png")
            
            if viz_path and os.path.exists(viz_path):
                print(f"✅ Visualization created: {viz_path}")
            else:
                print("❌ Visualization failed")
                return False
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            print("\n🎉 Streamlit workflow test completed successfully!")
            return True
        else:
            print("❌ Detection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streamlit_workflow()
    if success:
        print("\n✅ All tests passed! Streamlit app should work correctly.")
    else:
        print("\n❌ Tests failed!")