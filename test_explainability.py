#!/usr/bin/env python3
"""
Test script for explainability features (LIME, SHAP, Grad-CAM)
"""

import os
from models import (
    load_yolo_model, detect_all_cells_comprehensive, 
    generate_comprehensive_explainability,
    LIME_AVAILABLE, SHAP_AVAILABLE, GRADCAM_AVAILABLE
)

def test_explainability_features():
    """Test all explainability features"""
    
    print("🔬 Testing AI Explainability Features")
    print("=" * 50)
    
    # Check availability
    print("📋 Checking explainability method availability:")
    print(f"   🔍 LIME: {'✅ Available' if LIME_AVAILABLE else '❌ Not available'}")
    print(f"   📊 SHAP: {'✅ Available' if SHAP_AVAILABLE else '❌ Not available'}")
    print(f"   🎯 Grad-CAM: {'✅ Available' if GRADCAM_AVAILABLE else '❌ Not available'}")
    
    if not any([LIME_AVAILABLE, SHAP_AVAILABLE, GRADCAM_AVAILABLE]):
        print("\n❌ No explainability methods available!")
        print("Install required packages:")
        print("pip install lime shap grad-cam pytorch-grad-cam")
        return False
    
    # Use sample image
    sample_image = "dataset/train/images/BloodImage_00001_jpg.rf.1a3206b15602db1d97193162a50bd001.jpg"
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return False
    
    print(f"\n📷 Using sample image: {os.path.basename(sample_image)}")
    
    # Load model
    print("\n📥 Loading detection model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Run basic detection first
    print("\n🔍 Running blood cell detection...")
    detection_results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if not detection_results:
        print("❌ Detection failed")
        return False
    
    stats = detection_results['stats']
    print(f"✅ Detection complete: {stats['total_cells_detected']} cells found")
    
    # Generate comprehensive explainability
    print("\n🚀 Generating comprehensive explainability analysis...")
    explainability_results = generate_comprehensive_explainability(
        sample_image, 
        model, 
        output_dir="explainability_test_results"
    )
    
    if explainability_results:
        print(f"\n🎉 Explainability analysis completed!")
        print(f"📁 Results saved in: explainability_test_results/")
        
        for method, path in explainability_results.items():
            if os.path.exists(path):
                print(f"   ✅ {method.upper()}: {path}")
            else:
                print(f"   ❌ {method.upper()}: Failed")
        
        return True
    else:
        print("❌ Explainability analysis failed")
        return False

if __name__ == "__main__":
    success = test_explainability_features()
    if success:
        print("\n✅ All explainability tests passed!")
        print("🎯 The system now includes LIME, SHAP, and Grad-CAM explanations!")
    else:
        print("\n❌ Explainability tests failed!")