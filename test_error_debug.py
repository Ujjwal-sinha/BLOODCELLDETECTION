#!/usr/bin/env python3
"""
Debug the Platelets_count error
"""

import os
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def debug_platelets_error():
    """Debug the specific error"""
    
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print("❌ Sample image not found")
        return
    
    # Load model
    model = load_yolo_model('yolo11n.pt')
    
    # Run detection
    print("🔍 Running detection...")
    results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if results:
        print("✅ Detection successful")
        stats = results['stats']
        
        # Print all keys in stats
        print("📊 Stats keys:", list(stats.keys()))
        
        # Check specific keys
        for key in ['RBC_count', 'WBC_count', 'Platelet_count', 'Platelets_count']:
            if key in stats:
                print(f"✅ {key}: {stats[key]}")
            else:
                print(f"❌ {key}: NOT FOUND")
        
        # Try visualization
        print("\n🎨 Testing visualization...")
        try:
            viz_path = visualize_all_cells(sample_image, results, output_path="test_error_viz.png")
            if viz_path:
                print(f"✅ Visualization successful: {viz_path}")
            else:
                print("❌ Visualization failed")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Detection failed")

if __name__ == "__main__":
    debug_platelets_error()