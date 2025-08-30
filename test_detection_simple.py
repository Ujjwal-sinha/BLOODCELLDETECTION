#!/usr/bin/env python3
"""
Simple test for blood cell detection
"""

import os
from models import load_yolo_model, detect_all_cells_comprehensive, create_fallback_detection

def test_detection():
    """Test blood cell detection"""
    
    print("🔬 Testing Blood Cell Detection")
    print("=" * 50)
    
    # Try to load YOLO model
    print("📥 Loading YOLO model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("⚠️ YOLO model not available, will use computer vision fallback")
    else:
        print("✅ YOLO model loaded successfully")
    
    # Look for sample images
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
        print("Please add some blood smear images to test with")
        return False
    
    print(f"📷 Testing with sample image: {sample_image}")
    
    # Test detection
    print("\n🔍 Running detection test...")
    detection_results = detect_all_cells_comprehensive(model, sample_image, confidence_threshold=0.01)
    
    if detection_results:
        stats = detection_results['stats']
        print(f"\n✅ Detection successful!")
        print(f"   Total cells detected: {stats['total_cells_detected']}")
        print(f"   RBC: {stats['RBC_count']}")
        print(f"   WBC: {stats['WBC_count']}")
        print(f"   Platelets: {stats['Platelet_count']}")
        print(f"   Detection method: {detection_results.get('detection_method', 'Unknown')}")
        
        if stats['total_cells_detected'] > 0:
            print(f"   Overall confidence: {stats['confidence_scores']['Overall']:.2%}")
            print(f"   Detection density: {stats['detection_density']:.6f} cells/pixel")
            return True
        else:
            print("⚠️ No cells detected - this might indicate an issue with the image or model")
            return False
    else:
        print("❌ Detection failed completely")
        return False

if __name__ == "__main__":
    success = test_detection()
    if success:
        print("\n🎉 Detection test passed!")
    else:
        print("\n❌ Detection test failed!")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have blood smear images in the dataset folders")
        print("2. Check that the images are clear and contain visible blood cells")
        print("3. The system will use computer vision fallback if YOLO model isn't trained")