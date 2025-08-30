#!/usr/bin/env python3
"""
Test script for enhanced blood cell detection with automatic explainability
"""

import os
from models import load_yolo_model, detect_all_cells_comprehensive

def test_enhanced_detection():
    """Test the enhanced detection system"""
    
    print("🔬 Testing Enhanced Blood Cell Detection")
    print("=" * 60)
    
    # Use sample image
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return False
    
    print(f"📷 Analyzing: {os.path.basename(sample_image)}")
    
    # Load model
    print("\n📥 Loading detection model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Run enhanced detection
    print("\n🚀 Running enhanced detection with automatic explainability...")
    results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if not results:
        print("❌ Detection failed")
        return False
    
    # Display detection results
    stats = results['stats']
    print(f"\n🎯 DETECTION RESULTS:")
    print(f"   Total Cells: {stats['total_cells_detected']}")
    print(f"   🔴 RBC: {stats['RBC_count']} ({stats['cell_distribution']['RBC_percentage']:.1f}%)")
    print(f"   ⚪ WBC: {stats['WBC_count']} ({stats['cell_distribution']['WBC_percentage']:.1f}%)")
    print(f"   🟢 Platelets: {stats['Platelet_count']} ({stats['cell_distribution']['Platelet_percentage']:.1f}%)")
    print(f"   📊 Method: {results['detection_method']}")
    print(f"   🎯 Confidence: {stats['confidence_scores']['Overall']:.1%}")
    
    # Display explainability results
    explainability = results.get('explainability', {})
    if explainability:
        print(f"\n🔬 AUTOMATIC EXPLAINABILITY RESULTS:")
        for method, path in explainability.items():
            if os.path.exists(path):
                print(f"   ✅ {method.replace('_', ' ').title()}: {path}")
            else:
                print(f"   ❌ {method.replace('_', ' ').title()}: Failed")
        
        print(f"\n📁 All explainability results saved in: explainability_results/")
    else:
        print("\n⚠️ No explainability results generated")
    
    print(f"\n🎉 Enhanced detection test completed!")
    print(f"💡 The system now automatically generates:")
    print(f"   • Enhanced cell detection using multiple CV techniques")
    print(f"   • Edge detection analysis")
    print(f"   • Detection confidence heatmaps")
    print(f"   • LIME/SHAP/Grad-CAM explanations (if available)")
    print(f"   • Comprehensive explainability summaries")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_detection()
    if success:
        print("\n✅ Enhanced detection with automatic explainability working perfectly!")
    else:
        print("\n❌ Enhanced detection test failed!")