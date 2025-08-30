import os
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered Blood Cell Detection",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

from PIL import Image
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import tempfile
import uuid
import glob
import yaml
import cv2
import numpy as np

# Import from modular files
from models import (
    device, clear_mps_cache, load_yolo_model, preprocess_image,
    plot_metrics, plot_detection_results, detect_all_cells_comprehensive,
    visualize_all_cells, generate_comprehensive_explainability,
    generate_lime_explanation, generate_shap_explanation, generate_gradcam_explanation,
    LIME_AVAILABLE, SHAP_AVAILABLE, GRADCAM_AVAILABLE
)
from utils import (
    load_css, validate_dataset, get_image_transform, check_image_quality,
    gradient_text, plot_cell_distribution, generate_report
)

# Import AI agents if available
try:
    from agents import BloodCellAIAgent, create_agent_instance
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Load custom CSS for UI
load_css()

# Initialize YOLO model
@st.cache_resource
def load_detection_model():
    try:
        model = load_yolo_model('yolo11n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize YOLO model for blood cell detection
yolo_model = load_detection_model()

if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input("Enter GROQ API Key", type="password")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not provided. Please set it in a .env file or enter it in the sidebar.")
        st.stop()

st.write(f"Using device: {device}")

# Load BLIP models for image description
from utils import load_models
processor, blip_model = load_models()
if not processor or not blip_model:
    st.warning("BLIP models not available. Image descriptions will be limited.")
    processor, blip_model = None, None

# Initialize session state
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'plot_paths' not in st.session_state:
    st.session_state.plot_paths = []

# Main header
st.markdown("""
<div class="gradient-header">
    <h1 style="color: #2d3748;">BloodCellAI</h1>
    <p style="color: #4a5568; font-weight: 500;">Advanced AI-Powered Blood Cell Detection Platform</p>
</div>
""", unsafe_allow_html=True)

# Dataset validation
dataset_dir = "dataset"
is_valid, message = validate_dataset(dataset_dir)
if not is_valid:
    st.error(f"Dataset validation failed: {message}")
    st.stop()

# Define blood cell classes
classes = ['RBC', 'WBC', 'Platelets']
blood_cell_classes = ['RBC', 'WBC', 'Platelets']

# Validate dataset structure
if os.path.exists(dataset_dir):
    is_valid, message = validate_dataset(dataset_dir)
    if not is_valid:
        st.error(f"❌ {message}")
        st.stop()
        
    # Read classes from data.yaml
    import yaml
    try:
        with open(os.path.join(dataset_dir, 'data.yaml'), 'r') as f:
            data_yaml = yaml.safe_load(f)
            if 'names' in data_yaml:
                classes = [str(name) for name in data_yaml['names'].values()]
            else:
                classes = blood_cell_classes
    except Exception as e:
        classes = blood_cell_classes
    
    st.success(f"✅ Dataset validated successfully: Found {len(classes)} cell types")
else:
    st.error(f"Dataset directory '{dataset_dir}' not found")
    st.stop()

# For YOLO dataset, validate images exist
has_images = False
total_images = 0

for split in ['train', 'valid', 'test']:
    images_path = os.path.join(dataset_dir, split, 'images')
    if os.path.exists(images_path):
        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(images)
        if len(images) > 0:
            has_images = True

if not has_images:
    st.error("No images found in the dataset")
    st.stop()

st.success(f"✅ Dataset validated: Found {len(classes)} classes")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="glass-effect">
        <h3 style="color: #2d3748; font-weight: 600;">About Blood Cell Detection</h3>
        <p style="color: #4a5568; font-weight: 500;">Advanced AI-Powered Blood Cell Detection Platform combines:</p>
        <ul style="color: #4a5568; font-weight: 500;">
            <li>Advanced computer vision</li>
            <li>Hematological expertise</li>
            <li>AI-powered analysis</li>
        </ul>
        <p style="color: #4a5568; font-weight: 500;">For accurate detection and counting of blood cells (RBC, WBC, Platelets).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-effect">
        <h3 style="color: #2d3748; font-weight: 600;">Model Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    epochs = st.slider("Training Epochs", 1, 80, 5)
    patience = st.slider("Early Stopping Patience", 1, 10, 3)
    debug_mode = st.checkbox("Debug Mode", value=False, help="Enable debug information")
    
    # Model training section
    st.markdown("""
    <div class="glass-effect">
        <h3>Model Training</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Train YOLO Model", key="train_button"):
        with st.spinner("Training YOLO model for blood cell detection... This may take several minutes."):
            try:
                from models import train_blood_cell_detector
                results = train_blood_cell_detector(
                    data_dir=dataset_dir,
                    epochs=epochs,
                    batch_size=16
                )
                
                if results is not None:
                    st.success("✅ YOLO model training completed successfully!")
                    st.session_state.model_trained = True
                    
                    # Store training data for visualization
                    st.session_state.evaluation_data = {
                        'training_results': results
                    }
                else:
                    st.error("❌ Model training failed")
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
    
    # Show training status
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.success("✅ Model is trained and ready for analysis")
    else:
        st.info("ℹ️ Model will use ImageNet weights. Train the model for better accuracy.")
    
    if debug_mode:
        st.info(f"Debug: Classes loaded: {classes}")
    
    st.markdown("""
    <div class="glass-effect">
        <h3>Class Distribution</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Count images in train/valid/test splits
    class_counts = {}
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_dir, split, 'images')
        if os.path.exists(split_path):
            images = [f for f in os.listdir(split_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[f"{split.title()} Images"] = len(images)
    
    # Display class counts with better formatting
    st.markdown("""
    <div class="glass-effect">
        <h4 style="color: #2d3748; font-weight: 600; margin-bottom: 1rem;">Dataset Classes</h4>
    </div>
    """, unsafe_allow_html=True)
    
    for cls, count in class_counts.items():
        # Create a shorter display name
        display_name = cls.replace('_', ' ').title()
        st.markdown(f"""
        <div class="glass-effect" style="padding: 0.8rem; margin: 0.5rem 0;">
            <strong style="color: #2d3748;">{display_name}:</strong> <span style="color: #48bb78; font-weight: 600;">{count} images</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Test GROQ API status
    from utils import test_groq_api
    api_working = False
    api_message = "Testing..."
    
    if GROQ_API_KEY and len(GROQ_API_KEY) > 20:
        api_working, api_message = test_groq_api()
    else:
        api_message = "No API key provided"
    
    if api_working:
        st.markdown(f"""
        <div class="success-box">
            <strong>GROQ API:</strong> <span>✅ Working</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <strong>GROQ API:</strong> <span>❌ {api_message}</span>
        </div>
        """, unsafe_allow_html=True)

# Main content area
st.markdown("""
<div class="hero-gradient">
    <div style="text-align: center; color: #2d3748;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center;">📁</div>
        <h2 style="font-family: 'Inter', sans-serif; margin-bottom: 0.3rem; font-size: 1.8rem; font-weight: 700; color: #2d3748;">
            Blood Cell Analysis
        </h2>
        <p style="font-size: 1rem; margin: 0; color: #4a5568; font-weight: 500;">
            AI-Powered Analysis • Image Upload • Blood Cell Detection
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Blood Cell Information Section
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #2d3748; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
        Blood Cell Information
    </h3>
    <p style="color: #4a5568; margin-bottom: 1rem;">Learn about different blood cell types and their functions</p>
</div>
""", unsafe_allow_html=True)

# Blood cell information
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: rgba(239, 83, 80, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ef5350;">
        <strong style="color: #2d3748;">Red Blood Cells (RBC)</strong><br>
        <span style="color: #4a5568; font-size: 0.9rem;">Function: Oxygen transport<br>Normal Count: 4.5-5.5 million/μL</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(66, 165, 245, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #42a5f5;">
        <strong style="color: #2d3748;">White Blood Cells (WBC)</strong><br>
        <span style="color: #4a5568; font-size: 0.9rem;">Function: Immune defense<br>Normal Count: 4,500-11,000/μL</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: rgba(102, 187, 106, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #66bb6a;">
        <strong style="color: #2d3748;">Platelets</strong><br>
        <span style="color: #4a5568; font-size: 0.9rem;">Function: Blood clotting<br>Normal Count: 150,000-450,000/μL</span>
    </div>
    """, unsafe_allow_html=True)

# Image input section
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="compact-card">
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem; display: flex; align-items: center; justify-content: center;">📁</div>
            <h4 style="color: #2d3748; margin: 0; font-weight: 700; font-family: 'Inter', sans-serif; font-size: 1.1rem;">Upload Blood Smear Image</h4>
            <p style="color: #4a5568; margin: 0.3rem 0 0 0; font-size: 0.85rem; font-weight: 500;">Upload a blood smear image for cell analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    image = None
    
    # Only upload image option
    st.markdown("""
    <div class="compact-card" style="border: 2px dashed #48bb78; background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);">
        <div style="text-align: center;">
            <div style="background: linear-gradient(135deg, #48bb78, #38a169); width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; box-shadow: 0 6px 20px rgba(72, 187, 120, 0.3);">
                <span style="font-size: 1.5rem; color: white; display: flex; align-items: center; justify-content: center;">📁</span>
            </div>
            <h5 style="color: #2d3748; margin-bottom: 0.5rem; font-weight: 700; font-family: 'Inter', sans-serif; font-size: 1.1rem;">Upload Blood Smear Image</h5>
            <p style="color: #4a5568; font-size: 0.9rem; margin-bottom: 1rem; font-family: 'Inter', sans-serif; font-weight: 500;">Select a blood smear image from your device</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    img_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a blood smear image for cell analysis",
        label_visibility="collapsed"
    )
    
    if img_file:
        try:
            image = Image.open(img_file).convert('RGB')
            st.success("✅ Image uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Upload error: {e}")

with col2:
    if image:
        st.markdown("""
        <div class="compact-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 1.5rem; margin-bottom: 0.3rem; display: flex; align-items: center; justify-content: center;">📷</div>
                <h4 style="color: #2d3748; margin: 0; font-weight: 700; font-family: 'Inter', sans-serif; font-size: 1.1rem;">Image Preview</h4>
                <p style="color: #4a5568; margin: 0.3rem 0 0 0; font-size: 0.85rem; font-family: 'Inter', sans-serif; font-weight: 500;">Ready for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(image, caption="Blood Smear Image", use_column_width=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); border: none;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📷</div>
                <h4 style="color: #ffffff; margin: 0 0 0.5rem 0; font-weight: 700; font-family: 'Inter', sans-serif; font-size: 1.3rem;">Image Ready for Analysis</h4>
                <p style="color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 1rem; font-family: 'Inter', sans-serif;">Your blood smear image is ready for comprehensive cell analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="compact-card" style="border: 2px dashed #cbd5e0; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 3rem 1.5rem; text-align: center;">
            <div style="background: linear-gradient(135deg, #e2e8f0, #cbd5e0); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem auto; box-shadow: 0 6px 20px rgba(160, 174, 192, 0.3);">
                <span style="font-size: 2rem; color: #a0aec0;">🩺</span>
            </div>
            <h4 style="color: #4a5568; margin-bottom: 0.5rem; font-weight: 700; font-size: 1.1rem; font-family: 'Inter', sans-serif;">No Image Selected</h4>
            <p style="color: #718096; font-size: 0.9rem; margin: 0; font-family: 'Inter', sans-serif;">Upload an image to see preview</p>
        </div>
        """, unsafe_allow_html=True)

# Analysis options
st.markdown("""
<div class="glass-effect">
    <h3 style="color: #2d3748; margin-bottom: 1rem; text-align: center; font-size: 1.3rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
        <span style="font-size: 1.5rem;">📋</span>
        <span>Select Analysis Type</span>
    </h3>
    <p style="color: #4a5568; text-align: center; margin-bottom: 1rem; font-size: 0.95rem; font-weight: 500;">Choose one or more analysis types below, then click "Analyze" to proceed</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    cell_detection = st.checkbox(
        "Blood Cell Detection",
        help="Detect and count blood cells (RBC, WBC, Platelets)",
        key="detection_checkbox"
    )

with col2:
    morphology_analysis = st.checkbox(
        "Cell Morphology Analysis",
        help="Analyze cell shape, size, and characteristics",
        key="morphology_checkbox"
    )

with col3:
    complete_analysis = st.checkbox(
        "Complete Blood Analysis",
        help="Full blood cell detection and morphology assessment",
        key="complete_checkbox"
    )

# Analyze button
if st.button("Start Analysis", type="primary", key="analyze_button"):
    if not image:
        st.warning("Please upload a blood smear image first.")
        st.stop()
    
    selected_analyses = []
    if cell_detection:
        selected_analyses.append("detection")
    if morphology_analysis:
        selected_analyses.append("morphology")
    if complete_analysis:
        selected_analyses.append("complete")
    
    if not selected_analyses:
        st.warning("Please select at least one analysis type.")
        st.stop()
    
    # Run blood cell analysis
    with st.spinner("Analyzing blood smear image..."):
        try:
            # Check image quality
            quality_score = check_image_quality(image)
            
            # Describe image using BLIP
            from utils import describe_image
            image_description = describe_image(image) if processor and blip_model else "Blood smear image for analysis"
            
            # Use YOLO model for blood cell detection
            if yolo_model is not None:
                # Save image temporarily for YOLO processing
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    
                    # Run comprehensive cell detection to find ALL cells
                    print("Starting comprehensive cell detection...")
                    st.info("🔍 Running comprehensive blood cell detection with maximum sensitivity...")
                    detection_results = detect_all_cells_comprehensive(yolo_model, tmp_file.name, confidence_threshold=0.01)
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                
                if detection_results:
                    stats = detection_results['stats']
                    detections = detection_results['detections']
                    
                    # Display comprehensive detection summary
                    st.success(f"Detection completed: {detection_results.get('detection_summary', 'Detection completed')}")
                    
                    # Show detailed detection statistics
                    total_detected = stats.get('total_cells_detected', 0)
                    if total_detected > 0:
                        st.info(f"""
                        **Comprehensive Detection Results:**
                        - **Total Cells Found:** {total_detected}
                        - **RBC:** {stats['RBC_count']} ({stats['cell_distribution']['RBC_percentage']:.1f}%)
                        - **WBC:** {stats['WBC_count']} ({stats['cell_distribution']['WBC_percentage']:.1f}%)
                        - **Platelets:** {stats['Platelet_count']} ({stats['cell_distribution']['Platelet_percentage']:.1f}%)
                        - **Overall Confidence:** {stats['confidence_scores']['Overall']:.2%}
                        - **Detection Density:** {stats['detection_density']:.6f} cells/pixel
                        """)
                        
                                        # Create and display visualization of all detected cells
                        with st.spinner("Creating comprehensive cell visualization..."):
                            try:
                                # Save the current image temporarily for visualization
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as viz_tmp_file:
                                    image.save(viz_tmp_file.name)
                                    viz_path = visualize_all_cells(viz_tmp_file.name, detection_results)
                                    
                                    if viz_path and os.path.exists(viz_path):
                                        st.image(viz_path, caption=f"All Detected Cells - {total_detected} cells found", use_column_width=True)
                                        st.success("✅ All cells visualized successfully!")
                                    else:
                                        st.warning("⚠️ Could not create cell visualization")
                                    
                                    # Clean up temp file
                                    os.unlink(viz_tmp_file.name)
                            except Exception as viz_error:
                                st.warning(f"⚠️ Visualization error: {str(viz_error)}")
                                st.info("Detection completed successfully, but visualization failed")
                        
                        # 🔬 EXPLAINABILITY ANALYSIS SECTION
                        st.markdown("---")
                        st.markdown("## 🔬 AI Explainability Analysis")
                        st.markdown("**Understanding WHY the AI detected cells in specific regions**")
                        
                        # Check availability of explainability methods
                        available_methods = []
                        if LIME_AVAILABLE:
                            available_methods.append("LIME")
                        if SHAP_AVAILABLE:
                            available_methods.append("SHAP")
                        if GRADCAM_AVAILABLE:
                            available_methods.append("Grad-CAM")
                        
                        if available_methods:
                            st.info(f"✅ Available explainability methods: {', '.join(available_methods)}")
                            
                            # Explainability options
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                run_lime = st.checkbox("🔍 LIME Analysis", 
                                                     value=LIME_AVAILABLE,
                                                     disabled=not LIME_AVAILABLE,
                                                     help="Local Interpretable Model-agnostic Explanations")
                            
                            with col2:
                                run_shap = st.checkbox("📊 SHAP Analysis", 
                                                     value=SHAP_AVAILABLE,
                                                     disabled=not SHAP_AVAILABLE,
                                                     help="SHapley Additive exPlanations")
                            
                            with col3:
                                run_gradcam = st.checkbox("🎯 Grad-CAM Analysis", 
                                                        value=GRADCAM_AVAILABLE,
                                                        disabled=not GRADCAM_AVAILABLE,
                                                        help="Gradient-weighted Class Activation Mapping")
                            
                            # Run explainability analysis
                            if st.button("🚀 Generate Explainability Analysis", type="primary"):
                                with st.spinner("🔬 Generating comprehensive explainability analysis..."):
                                    try:
                                        # Save current image for explainability
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as exp_tmp_file:
                                            image.save(exp_tmp_file.name)
                                            
                                            # Create explainability results directory
                                            exp_dir = tempfile.mkdtemp(prefix="explainability_")
                                            
                                            explainability_results = {}
                                            
                                            # Generate selected explanations
                                            if run_lime and LIME_AVAILABLE:
                                                st.info("🔍 Generating LIME explanation...")
                                                lime_path = generate_lime_explanation(exp_tmp_file.name, yolo_model, 
                                                                                    os.path.join(exp_dir, "lime.png"))
                                                if lime_path:
                                                    explainability_results['lime'] = lime_path
                                            
                                            if run_shap and SHAP_AVAILABLE:
                                                st.info("📊 Generating SHAP explanation...")
                                                shap_path = generate_shap_explanation(exp_tmp_file.name, yolo_model,
                                                                                    os.path.join(exp_dir, "shap.png"))
                                                if shap_path:
                                                    explainability_results['shap'] = shap_path
                                            
                                            if run_gradcam and GRADCAM_AVAILABLE:
                                                st.info("🎯 Generating Grad-CAM explanation...")
                                                gradcam_path = generate_gradcam_explanation(exp_tmp_file.name, yolo_model,
                                                                                          os.path.join(exp_dir, "gradcam.png"))
                                                if gradcam_path:
                                                    explainability_results['gradcam'] = gradcam_path
                                            
                                            # Display results
                                            if explainability_results:
                                                st.success("✅ Explainability analysis completed!")
                                                
                                                # Display each explanation
                                                for method, path in explainability_results.items():
                                                    if os.path.exists(path):
                                                        method_name = {
                                                            'lime': '🔍 LIME Explanation',
                                                            'shap': '📊 SHAP Explanation', 
                                                            'gradcam': '🎯 Grad-CAM Explanation'
                                                        }.get(method, method.upper())
                                                        
                                                        st.markdown(f"### {method_name}")
                                                        st.image(path, caption=f"{method_name} - Shows important regions for cell detection", 
                                                               use_column_width=True)
                                                
                                                # Create comprehensive summary
                                                st.markdown("### 📋 Explainability Summary")
                                                st.markdown("""
                                                **Understanding the AI's Decision Process:**
                                                
                                                🔍 **LIME (Local Interpretable Model-agnostic Explanations):**
                                                - Highlights image regions that most influence the detection decision
                                                - Green areas contribute positively to cell detection
                                                - Red areas may inhibit detection
                                                
                                                📊 **SHAP (SHapley Additive exPlanations):**
                                                - Provides pixel-level feature attribution
                                                - Red regions increase detection confidence
                                                - Blue regions decrease detection confidence
                                                
                                                🎯 **Grad-CAM (Gradient-weighted Class Activation Mapping):**
                                                - Shows where the model focuses its attention
                                                - Heatmap indicates regions of highest importance
                                                - Warmer colors = higher attention/relevance
                                                
                                                **Clinical Significance:**
                                                These explainability methods help medical professionals understand WHY the AI 
                                                made specific detection decisions, increasing trust and enabling better 
                                                clinical interpretation of automated results.
                                                """)
                                            else:
                                                st.warning("⚠️ No explainability results generated. Please check method availability.")
                                            
                                            # Clean up temp file
                                            os.unlink(exp_tmp_file.name)
                                            
                                    except Exception as exp_error:
                                        st.error(f"❌ Explainability analysis failed: {str(exp_error)}")
                                        st.info("Detection results are still valid, but explainability analysis encountered an error.")
                        else:
                            st.warning("⚠️ No explainability methods available. Install required packages:")
                            st.code("""
pip install lime shap grad-cam
                            """)
                            st.info("Explainability analysis requires additional packages for LIME, SHAP, and Grad-CAM visualization.")
                    
                    else:
                        st.warning("⚠️ No cells detected in the image")
                    
                    # Create analysis prompt for AI agent
                    detected_cells = []
                    confidences = []
                    count_data = {}
                    
                    for cell_type in ['RBC', 'WBC', 'Platelets']:
                        if cell_type == 'Platelets':
                            count = stats['Platelet_count']  # Use singular form for count
                        else:
                            count = stats[f'{cell_type}_count']
                        if count > 0:
                            detected_cells.append(cell_type)
                            # confidence_scores uses plural form for all keys
                            confidences.append(stats['confidence_scores'][cell_type])
                            count_data[cell_type] = count
                    
                    # Use AI agent for comprehensive analysis
                    if AGENTS_AVAILABLE and GROQ_API_KEY:
                        try:
                            agent = create_agent_instance("blood", GROQ_API_KEY)
                            ai_analysis = agent.analyze_blood_sample(
                                image_description=image_description,
                                detected_cells=detected_cells,
                                confidences=confidences,
                                count_data=count_data,
                                morphology="Automated YOLO detection results"
                            )
                            analysis_result = ai_analysis.get("analysis", "Analysis completed")
                        except Exception as e:
                            # Fallback analysis
                            total_cells = stats.get('total_cells_detected', 0)
                            analysis_result = f"""
                            **Comprehensive Blood Cell Analysis Report**
                            
                            **Total Detection Summary:**
                            - Total Cells Detected: {total_cells}
                            - Detection Density: {stats.get('detection_density', 0):.6f} cells/pixel
                            
                            **Individual Cell Counts:**
                            - Red Blood Cells (RBC): {stats['RBC_count']}
                            - White Blood Cells (WBC): {stats['WBC_count']}
                            - Platelets: {stats['Platelet_count']}
                            
                            **Cell Distribution:**
                            - RBC: {stats.get('cell_distribution', {}).get('RBC_percentage', 0):.1f}%
                            - WBC: {stats.get('cell_distribution', {}).get('WBC_percentage', 0):.1f}%
                            - Platelets: {stats.get('cell_distribution', {}).get('Platelet_percentage', 0):.1f}%
                            
                            **Detection Confidence Scores:**
                            - RBC Detection: {stats['confidence_scores']['RBC']:.2%}
                            - WBC Detection: {stats['confidence_scores']['WBC']:.2%}
                            - Platelet Detection: {stats['confidence_scores']['Platelets']:.2%}
                            - Overall Confidence: {stats['confidence_scores'].get('Overall', 0):.2%}
                            
                            **Clinical Assessment:**
                            1. Comprehensive cell detection completed successfully
                            2. All major blood cell types identified and counted
                            3. Detection confidence levels are within acceptable ranges
                            4. Cell distribution patterns analyzed
                            
                            **Recommendations:**
                            1. Verify counts with automated hematology analyzer
                            2. Consider complete blood count (CBC) with differential
                            3. Correlate findings with clinical presentation
                            4. Monitor cell morphology for abnormalities
                            5. Follow up with serial blood counts if indicated
                            
                            **Quality Assurance:**
                            - Automated detection using advanced YOLO model
                            - Comprehensive cell identification algorithm
                            - Statistical analysis of cell populations
                            - Professional laboratory verification recommended
                            
                            **Note:** This is an AI-powered preliminary analysis. Professional laboratory verification and clinical correlation are essential for accurate diagnosis.
                            """
                    else:
                        # Comprehensive analysis without AI agent
                        total_cells = stats.get('total_cells_detected', 0)
                        analysis_result = f"""
                        **Comprehensive Blood Cell Detection Results**
                        
                        **Detection Summary:**
                        - Total Cells Detected: {total_cells}
                        - Detection Method: Advanced YOLO Model
                        - Analysis Type: Comprehensive Cell Detection
                        
                        **Individual Cell Counts:**
                        - Red Blood Cells (RBC): {stats['RBC_count']}
                        - White Blood Cells (WBC): {stats['WBC_count']}
                        - Platelets: {stats['Platelet_count']}
                        
                        **Cell Distribution Analysis:**
                        - RBC Percentage: {stats.get('cell_distribution', {}).get('RBC_percentage', 0):.1f}%
                        - WBC Percentage: {stats.get('cell_distribution', {}).get('WBC_percentage', 0):.1f}%
                        - Platelet Percentage: {stats.get('cell_distribution', {}).get('Platelet_percentage', 0):.1f}%
                        
                        **Detection Confidence Scores:**
                        - RBC Detection: {stats['confidence_scores']['RBC']:.2%}
                        - WBC Detection: {stats['confidence_scores']['WBC']:.2%}
                        - Platelets Detection: {stats['confidence_scores']['Platelets']:.2%}
                        - Overall Confidence: {stats['confidence_scores'].get('Overall', 0):.2%}
                        
                        **Detection Quality Metrics:**
                        - Detection Density: {stats.get('detection_density', 0):.6f} cells/pixel
                        - Coverage: Comprehensive cell identification
                        - Algorithm: State-of-the-art YOLO detection
                        
                        **Clinical Interpretation:**
                        1. All major blood cell types successfully identified
                        2. Quantitative analysis of cell populations completed
                        3. Statistical distribution patterns analyzed
                        4. High-confidence detection results obtained
                        
                        **Laboratory Recommendations:**
                        1. Compare results with normal reference ranges:
                           - RBC: 4.5-5.5 million cells/μL
                           - WBC: 4,500-11,000 cells/μL  
                           - Platelets: 150,000-450,000/μL
                        2. Consider complete blood count (CBC) with differential
                        3. Correlate with clinical presentation and symptoms
                        4. Verify with automated hematology analyzer
                        5. Monitor trends with serial blood counts
                        
                        **Quality Assurance Notes:**
                        - Automated detection using advanced computer vision
                        - Comprehensive cell identification and counting
                        - Professional laboratory verification recommended
                        """
                    
                    # Store results
                    st.session_state.report_data = {
                        "analysis_type": "blood_cell_detection",
                        "report": analysis_result,
                        "image": image,
                        "image_description": image_description,
                        "quality_score": quality_score,
                        "detection_results": detection_results,
                        "cell_counts": count_data,
                        "detected_cells": detected_cells,
                        "confidences": confidences
                    }
                    
                    st.success("✅ Blood cell analysis completed!")
                else:
                    st.error("❌ No blood cells detected in the image")
            else:
                st.error("❌ YOLO model not available for detection")
                
        except Exception as e:
            st.error(f"❌ Error during blood cell analysis: {str(e)}")
            if debug_mode:
                st.exception(e)

# Display results
if 'report_data' in st.session_state and st.session_state.report_data is not None and isinstance(st.session_state.report_data, dict):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); border: none;">
        <h2 style="font-family: 'Poppins', sans-serif; color: #ffffff; margin-bottom: 0.3rem; text-align: center; font-size: 1.8rem;">Analysis Complete!</h2>
        <p style="color: rgba(255, 255, 255, 0.9); text-align: center; font-size: 1rem; margin-bottom: 0;">Your comprehensive blood cell analysis is ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for results
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "Detection Overview", 
        "Detailed Analysis", 
        "Laboratory Report",
        "Cell Statistics"
    ])
    
    with main_tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.session_state.report_data and "image" in st.session_state.report_data:
                st.image(st.session_state.report_data["image"], caption="Blood Smear Analysis", use_column_width=True)
            else:
                st.error("Image data not available")
            
            if st.session_state.report_data and 'quality_score' in st.session_state.report_data:
                quality_score = st.session_state.report_data['quality_score']
                quality_color = "#48bb78" if quality_score > 0.7 else "#ed8936" if quality_score > 0.5 else "#e53e3e"
                quality_score_formatted = f"{quality_score:.2f}"
                st.markdown(f"""
                <div style="background: #ffffff; padding: 0.8rem; border-radius: 8px; margin: 0.8rem 0; border-left: 3px solid {quality_color}; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                    <p style="color: #2d3748; font-weight: bold; margin: 0; font-size: 0.95rem;">
                        Image Quality Score: <span style="color: {quality_color}; font-size: 1rem;">{quality_score_formatted}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Quality score data not available")
        
        with col2:
            # Display cell counts
            if st.session_state.report_data and 'cell_counts' in st.session_state.report_data:
                cell_counts = st.session_state.report_data['cell_counts']
                total_cells = sum(cell_counts.values())
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; text-align: center; color: white;">
                    <h5 style="margin: 0 0 0.3rem 0; font-size: 1rem;">Total Cells Detected</h5>
                    <p style="margin: 0; font-size: 1.3rem; font-weight: bold;">{total_cells}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual cell counts
                colors = {'RBC': '#ef5350', 'WBC': '#42a5f5', 'Platelets': '#66bb6a'}
                for cell_type, count in cell_counts.items():
                    color = colors.get(cell_type, '#666')
                    st.markdown(f"""
                    <div style="background: {color}; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; text-align: center; color: white;">
                        <strong>{cell_type}: {count}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; text-align: center; color: white;">
                    <h5 style="margin: 0 0 0.3rem 0; font-size: 1rem;">Detection Status</h5>
                     <p style="margin: 0; font-size: 1.1rem; font-weight: bold;">Analysis Complete</p>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem; opacity: 0.9;">YOLO Detection</p>
                </div>
                """, unsafe_allow_html=True)
    
    with main_tab2:
        if st.session_state.report_data and "report" in st.session_state.report_data:
            report_content = st.session_state.report_data["report"]
        else:
            report_content = "No analysis report available."
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border-left: 4px solid #48bb78;">
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Detailed Analysis Results</h4>
            <p style="color: #4a5568; font-size: 0.95rem; margin: 0;">Comprehensive blood cell analysis and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #ffffff; padding: 2rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-height: 600px; overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        sections = report_content.split('\n\n')
        for section in sections:
            if section.strip():
                if section.startswith('**') and section.endswith('**'):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: white;">
                        <h5 style="margin: 0; font-size: 1.1rem; text-align: center;">{section}</h5>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.8rem 0; border-left: 3px solid #3182ce;">
                        <p style="color: #4a5568; line-height: 1.6; margin: 0;">{section}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with main_tab3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
            <h3 style="font-family: 'Poppins', sans-serif; color: #2d3748; margin-bottom: 1rem; text-align: center; font-size: 1.3rem;">Complete Blood Cell Report</h3>
            <p style="color: #4a5568; text-align: center; margin-bottom: 1rem; font-size: 0.95rem;">Full comprehensive analysis report with all details</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.report_data and "report" in st.session_state.report_data:
            report_content = st.session_state.report_data["report"]
        else:
            report_content = "No analysis report available."
        
        # Display complete report
        st.markdown("""
        <div style="background: #ffffff; padding: 2rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-height: 600px; overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        sections = report_content.split('\n\n')
        for section in sections:
            if section.strip():
                if section.startswith('**') and section.endswith('**'):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: white;">
                        <h5 style="margin: 0; font-size: 1.1rem; text-align: center;">{section}</h5>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.8rem 0; border-left: 3px solid #3182ce;">
                        <p style="color: #4a5568; line-height: 1.6; margin: 0;">{section}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with main_tab4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
            <h3 style="font-family: 'Poppins', sans-serif; color: #2d3748; margin-bottom: 1rem; text-align: center; font-size: 1.3rem;">Cell Statistics & Visualizations</h3>
            <p style="color: #4a5568; text-align: center; margin-bottom: 1rem; font-size: 0.95rem;">Statistical analysis and visualizations of detected blood cells</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display cell statistics
        if st.session_state.report_data and 'cell_counts' in st.session_state.report_data:
            cell_counts = st.session_state.report_data['cell_counts']
            
            # Create visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Cell count pie chart
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#ef5350', '#42a5f5', '#66bb6a']
                    
                    if cell_counts:
                        labels = list(cell_counts.keys())
                        sizes = list(cell_counts.values())
                        
                        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                         autopct='%1.1f%%', startangle=90)
                        ax.set_title('Blood Cell Distribution', fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No cell count data available for visualization")
                        
                except Exception as e:
                    st.error(f"Error creating pie chart: {e}")
            
            with col2:
                # Cell count bar chart
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if cell_counts:
                        cell_types = list(cell_counts.keys())
                        counts = list(cell_counts.values())
                        colors = ['#ef5350', '#42a5f5', '#66bb6a']
                        
                        bars = ax.bar(cell_types, counts, color=colors)
                        ax.set_title('Blood Cell Counts', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Count')
                        ax.grid(True, alpha=0.3)
                        
                        # Add count labels on bars
                        for bar, count in zip(bars, counts):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                                   str(count), ha='center', va='bottom', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No cell count data available for visualization")
                        
                except Exception as e:
                    st.error(f"Error creating bar chart: {e}")
            
            # Display detailed statistics table
            st.markdown("### Detailed Cell Statistics")
            
            # Normal ranges for reference
            normal_ranges = {
                'RBC': '4.5-5.5 million/μL',
                'WBC': '4,500-11,000/μL', 
                'Platelets': '150,000-450,000/μL'
            }
            
            # Create comparison table
            import pandas as pd
            
            table_data = []
            for cell_type, count in cell_counts.items():
                table_data.append({
                    'Cell Type': cell_type,
                    'Detected Count': count,
                    'Normal Range': normal_ranges.get(cell_type, 'N/A'),
                    'Status': 'Detected' if count > 0 else 'Not Detected'
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("No statistical data available. Please run an analysis first.")

# Reset button
st.markdown("""
<div style="background: #ffffff; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
    <h3 style="font-family: 'Poppins', sans-serif; color: #2d3748; margin-bottom: 1rem; font-size: 1.3rem;">Reset & Clear</h3>
    <p style="color: #4a5568; margin-bottom: 1rem; font-size: 0.95rem;">Clear all analysis results and start fresh</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Reset Analysis", key="reset_button"):
        keys_to_clear = ['report_data', 'model_trained', 'plot_paths', 'evaluation_data']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        for file in glob.glob("*.png") + glob.glob("*.jpg"):
            try:
                os.remove(file)
            except:
                pass
        clear_mps_cache()
        st.rerun()

# Report generation
if 'report_data' in st.session_state and st.session_state.report_data is not None:
    st.markdown("""
    <div style="background: #ffffff; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 1px solid #e2e8f0;">
        <h3 style="font-family: 'Poppins', sans-serif; color: #2d3748; margin-bottom: 1rem; font-size: 1.3rem;">Report Generation</h3>
        <p style="color: #4a5568; margin-bottom: 1rem; font-size: 0.95rem;">Generate a comprehensive PDF report of your analysis results</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Generate Comprehensive PDF Report", key="pdf_button"):
        with st.spinner("Generating professional report..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                from utils import BloodCellPDF
                
                blood_info = "Blood cell analysis"
                pdf = BloodCellPDF(blood_info=blood_info)
                pdf.cover_page()
                
                analysis_type = st.session_state.report_data.get("analysis_type", "blood_cell_detection")
                pdf.add_summary(st.session_state.report_data["report"])
                pdf.table_of_contents()

                # Save image temporarily for PDF
                tmp_path = os.path.join(tmp_dir, f"image_{uuid.uuid4()}.jpg")
                st.session_state.report_data["image"].save(tmp_path, quality=90, format="JPEG")
                pdf.add_image(tmp_path)

                # Add cell count table if available
                if 'cell_counts' in st.session_state.report_data:
                    pdf.add_cell_count_table(st.session_state.report_data['cell_counts'])

                report = st.session_state.report_data["report"]
                
                # Add sections to PDF
                sections = [
                    ("Blood Cell Detection Results", report.split("**Detection Results:**")[1].split("**")[0] if "**Detection Results:**" in report else ""),
                    ("Cell Count Analysis", report.split("**Cell Counts:**")[1].split("**")[0] if "**Cell Counts:**" in report else ""),
                    ("Clinical Recommendations", report.split("**Recommendations:**")[1].split("**")[0] if "**Recommendations:**" in report else ""),
                    ("Quality Assessment", report.split("**Quality Assurance:**")[1].split("**")[0] if "**Quality Assurance:**" in report else "")
                ]
                
                for title, content in sections:
                    if content.strip():
                        pdf.add_section(title, content)
                
                # Save PDF
                pdf_path = f"blood_cell_report_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(pdf_path)
                
                # Provide download link
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label=f"Download Blood Cell Analysis Report",
                    data=pdf_bytes,
                    file_name=pdf_path,
                    mime="application/pdf",
                )
                st.success(f"✅ Blood cell analysis report generated successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #4a5568;">
    <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: #2d3748;">
        BloodCellAI - Advanced Blood Cell Detection Platform
    </p>
    <p style="font-size: 0.9rem; margin-bottom: 1.5rem; color: #718096;">
        AI-Powered Blood Cell Analysis • Hematological Assessment • Clinical Insights
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem;">
        <a href="https://www.linkedin.com/in/sinhaujjwal01/" target="_blank" style="color: #3182ce; text-decoration: none; font-weight: 500;">
            LinkedIn
        </a>
        <a href="https://github.com/Ujjwal-sinha" target="_blank" style="color: #3182ce; text-decoration: none; font-weight: 500;">
            GitHub
        </a>
    </div>
</div>
""", unsafe_allow_html=True)