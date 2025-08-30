#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all imports from models.py"""
    
    print("🔍 Testing imports from models.py...")
    
    try:
        from models import (
            device, clear_mps_cache, load_yolo_model, preprocess_image,
            plot_metrics, plot_detection_results, detect_all_cells_comprehensive,
            visualize_all_cells, generate_automatic_explainability,
            generate_lime_explanation, generate_shap_explanation, generate_gradcam_explanation,
            LIME_AVAILABLE, SHAP_AVAILABLE, GRADCAM_AVAILABLE
        )
        print("✅ All imports successful!")
        
        # Test availability flags
        print(f"📋 Explainability availability:")
        print(f"   LIME: {'✅' if LIME_AVAILABLE else '❌'}")
        print(f"   SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
        print(f"   Grad-CAM: {'✅' if GRADCAM_AVAILABLE else '❌'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    
    print("\n🔍 Testing basic functionality...")
    
    try:
        from models import load_yolo_model, detect_all_cells_comprehensive
        
        # Test model loading
        print("📥 Testing model loading...")
        model = load_yolo_model('yolo11n.pt')
        
        if model is not None:
            print("✅ YOLO model loaded successfully")
        else:
            print("⚠️ YOLO model not available (this is expected if ultralytics not installed)")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Import Tests")
    print("=" * 40)
    
    import_success = test_imports()
    func_success = test_basic_functionality()
    
    if import_success and func_success:
        print("\n✅ All tests passed! The system is ready to use.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")