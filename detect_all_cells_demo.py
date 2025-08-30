#!/usr/bin/env python3
"""
Demo script to detect ALL cells in blood smear images
Focus: Maximum cell detection with comprehensive analysis
"""

import os
import sys
from PIL import Image
import argparse

# Import our detection functions
from models import load_yolo_model, detect_all_cells_comprehensive, visualize_all_cells

def detect_all_cells_in_image(image_path, output_dir="detection_results"):
    """
    Detect all cells in a blood smear image with maximum sensitivity
    
    Args:
        image_path (str): Path to the blood smear image
        output_dir (str): Directory to save results
    """
    
    print("🔬 Blood Cell Detection - Find ALL Cells")
    print("=" * 60)
    print(f"📷 Image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    print("\n📥 Loading YOLO model...")
    model = load_yolo_model('yolo11n.pt')
    
    if model is None:
        print("❌ Failed to load YOLO model")
        return False
    
    print("✅ YOLO model loaded successfully")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Run comprehensive detection with very low threshold to find ALL cells
    print("\n🔍 Running comprehensive cell detection...")
    print("   🎯 Using low confidence threshold (0.05) to detect maximum cells")
    print("   📊 Analyzing all cell types: RBC, WBC, Platelets")
    
    detection_results = detect_all_cells_comprehensive(
        model, 
        image_path, 
        confidence_threshold=0.05  # Very low threshold for maximum detection
    )
    
    if not detection_results:
        print("❌ Detection failed")
        return False
    
    # Display comprehensive results
    stats = detection_results['stats']
    detections = detection_results['detections']
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE DETECTION RESULTS")
    print("="*60)
    
    print(f"🎯 TOTAL CELLS DETECTED: {stats['total_cells_detected']}")
    print()
    
    print("📋 Individual Cell Counts:")
    print(f"   🔴 Red Blood Cells (RBC): {stats['RBC_count']}")
    print(f"   ⚪ White Blood Cells (WBC): {stats['WBC_count']}")
    print(f"   🟢 Platelets: {stats['Platelet_count']}")
    print()
    
    print("📊 Cell Distribution:")
    print(f"   🔴 RBC: {stats['cell_distribution']['RBC_percentage']:.1f}%")
    print(f"   ⚪ WBC: {stats['cell_distribution']['WBC_percentage']:.1f}%")
    print(f"   🟢 Platelets: {stats['cell_distribution']['Platelet_percentage']:.1f}%")
    print()
    
    print("🎯 Detection Quality:")
    print(f"   Overall Confidence: {stats['confidence_scores']['Overall']:.2%}")
    print(f"   RBC Confidence: {stats['confidence_scores']['RBC']:.2%}")
    print(f"   WBC Confidence: {stats['confidence_scores']['WBC']:.2%}")
    print(f"   Platelet Confidence: {stats['confidence_scores']['Platelets']:.2%}")
    print()
    
    print("📏 Detection Metrics:")
    print(f"   Detection Density: {stats['detection_density']:.6f} cells/pixel")
    print(f"   Average Cell Area: {stats['size_analysis']['avg_area']:.1f} pixels²")
    print(f"   Coverage Areas: {stats['detection_coverage']} grid regions")
    
    # Create comprehensive visualization
    print("\n🎨 Creating comprehensive visualization...")
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    viz_path = os.path.join(output_dir, f"{base_name}_all_cells_detected.png")
    
    viz_result = visualize_all_cells(
        image_path, 
        detection_results, 
        output_path=viz_path,
        show_confidence=True
    )
    
    if viz_result and os.path.exists(viz_result):
        print(f"✅ Comprehensive visualization saved: {viz_result}")
    else:
        print("⚠️ Could not create visualization")
    
    # Save detection summary
    summary_path = os.path.join(output_dir, f"{base_name}_detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE BLOOD CELL DETECTION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Detection Method: {detection_results.get('detection_method', 'YOLO')}\n")
        f.write(f"Confidence Threshold: {detection_results.get('confidence_threshold_used', 0.05)}\n\n")
        
        f.write("DETECTION RESULTS:\n")
        f.write(f"Total Cells Detected: {stats['total_cells_detected']}\n")
        f.write(f"RBC Count: {stats['RBC_count']} ({stats['cell_distribution']['RBC_percentage']:.1f}%)\n")
        f.write(f"WBC Count: {stats['WBC_count']} ({stats['cell_distribution']['WBC_percentage']:.1f}%)\n")
        f.write(f"Platelet Count: {stats['Platelet_count']} ({stats['cell_distribution']['Platelet_percentage']:.1f}%)\n\n")
        
        f.write("QUALITY METRICS:\n")
        f.write(f"Overall Confidence: {stats['confidence_scores']['Overall']:.2%}\n")
        f.write(f"Detection Density: {stats['detection_density']:.6f} cells/pixel\n")
        f.write(f"Average Cell Area: {stats['size_analysis']['avg_area']:.1f} pixels²\n")
    
    print(f"📄 Detection summary saved: {summary_path}")
    
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE CELL DETECTION COMPLETED!")
    print("="*60)
    
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Detect ALL cells in blood smear images")
    parser.add_argument("image_path", help="Path to blood smear image")
    parser.add_argument("--output", "-o", default="detection_results", 
                       help="Output directory for results (default: detection_results)")
    
    args = parser.parse_args()
    
    success = detect_all_cells_in_image(args.image_path, args.output)
    
    if success:
        print("\n✅ Detection completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Detection failed!")
        sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, try to find a sample image
    if len(sys.argv) == 1:
        print("🔍 No image specified, looking for sample images...")
        
        sample_dirs = ['dataset/train/images', 'dataset/valid/images', 'dataset/test/images']
        sample_image = None
        
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    sample_image = os.path.join(sample_dir, images[0])
                    break
        
        if sample_image:
            print(f"📷 Found sample image: {sample_image}")
            success = detect_all_cells_in_image(sample_image)
            sys.exit(0 if success else 1)
        else:
            print("❌ No sample images found. Please provide an image path.")
            print("Usage: python detect_all_cells_demo.py <image_path>")
            sys.exit(1)
    else:
        main()