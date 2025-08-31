#!/usr/bin/env python3
"""
Fast test script for optimized explainability methods
"""

import os
import time
import cv2
import numpy as np
from models import (
    generate_lime_explanation,
    generate_shap_explanation, 
    generate_gradcam_explanation,
    generate_edge_detection_analysis,
    detect_all_cells_comprehensive
)

def test_fast_explainability():
    """Test all explainability methods with speed measurements"""
    
    # Find a test image
    test_image = None
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'blood' in file.lower():
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        # Create a synthetic test image
        print("📸 Creating synthetic blood cell image for testing...")
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light background
        
        # Add some red circles (RBCs)
        cv2.circle(img, (100, 100), 20, (0, 0, 200), -1)  # Red in BGR
        cv2.circle(img, (200, 150), 25, (0, 0, 180), -1)
        cv2.circle(img, (300, 200), 18, (0, 0, 220), -1)
        
        # Add some blue circles (WBCs)
        cv2.circle(img, (150, 250), 30, (200, 100, 50), -1)  # Blue-ish
        cv2.circle(img, (250, 300), 28, (180, 120, 60), -1)
        
        # Add some small green dots (Platelets)
        cv2.circle(img, (80, 200), 8, (0, 150, 0), -1)  # Green
        cv2.circle(img, (320, 120), 6, (0, 180, 0), -1)
        cv2.circle(img, (180, 80), 7, (0, 160, 0), -1)
        
        test_image = 'synthetic_blood_test.jpg'
        cv2.imwrite(test_image, img)
        print(f"✅ Created synthetic test image: {test_image}")
    
    print(f"🧪 Testing explainability methods with: {test_image}")
    print("=" * 60)
    
    # Mock model for testing
    class MockModel:
        pass
    
    model = MockModel()
    
    # Test 1: LIME Explanation
    print("🔬 Testing LIME Explanation...")
    start_time = time.time()
    try:
        lime_result = generate_lime_explanation(test_image, model)
        lime_time = time.time() - start_time
        if lime_result:
            print(f"✅ LIME completed in {lime_time:.2f} seconds")
            print(f"   Output saved to: {lime_result}")
        else:
            print("❌ LIME failed")
    except Exception as e:
        print(f"❌ LIME error: {e}")
    
    # Test 2: SHAP Explanation  
    print("\n🔬 Testing SHAP Explanation...")
    start_time = time.time()
    try:
        shap_result = generate_shap_explanation(test_image, model)
        shap_time = time.time() - start_time
        if shap_result:
            print(f"✅ SHAP completed in {shap_time:.2f} seconds")
            print(f"   Output saved to: {shap_result}")
        else:
            print("❌ SHAP failed")
    except Exception as e:
        print(f"❌ SHAP error: {e}")
    
    # Test 3: Grad-CAM Explanation
    print("\n🔬 Testing Grad-CAM Explanation...")
    start_time = time.time()
    try:
        gradcam_result = generate_gradcam_explanation(test_image, model)
        gradcam_time = time.time() - start_time
        if gradcam_result:
            print(f"✅ Grad-CAM completed in {gradcam_time:.2f} seconds")
            print(f"   Output saved to: {gradcam_result}")
        else:
            print("❌ Grad-CAM failed")
    except Exception as e:
        print(f"❌ Grad-CAM error: {e}")
    
    # Test 4: Edge Detection Analysis
    print("\n🔬 Testing Edge Detection Analysis...")
    start_time = time.time()
    try:
        # First run detection to get results
        detection_results = detect_all_cells_comprehensive(model, test_image)
        if not detection_results:
            # Create mock detection results
            detection_results = {
                'detections': {
                    'RBC': [{'bbox': [80, 80, 120, 120], 'confidence': 0.9}],
                    'WBC': [{'bbox': [130, 230, 170, 270], 'confidence': 0.8}],
                    'Platelets': [{'bbox': [75, 195, 85, 205], 'confidence': 0.7}]
                },
                'stats': {
                    'total_cells_detected': 3,
                    'RBC_count': 1,
                    'WBC_count': 1,
                    'Platelet_count': 1
                }
            }
        
        edge_result = generate_edge_detection_analysis(test_image, detection_results, 'edge_analysis_test.png')
        edge_time = time.time() - start_time
        if edge_result:
            print(f"✅ Edge Detection completed in {edge_time:.2f} seconds")
            print(f"   Output saved to: {edge_result}")
        else:
            print("❌ Edge Detection failed")
    except Exception as e:
        print(f"❌ Edge Detection error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Fast Explainability Test Complete!")
    print("\nGenerated files:")
    for file in ['lime_explanation.png', 'shap_explanation.png', 'gradcam_explanation.png', 'edge_analysis_test.png']:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        elif os.path.exists(f"/tmp/{file}"):
            print(f"   ✅ /tmp/{file}")

if __name__ == "__main__":
    test_fast_explainability()