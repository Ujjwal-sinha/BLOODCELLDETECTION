import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import tempfile
import uuid
import glob
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered Blood Cell Detection",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)
import cv2
import numpy as np

# Import from modular files
from models import (
    device, clear_mps_cache, load_yolo_model, preprocess_image,
    plot_metrics, plot_detection_results, detect_all_cells_comprehensive,
    visualize_all_cells, enhance_cell_detection, create_cell_specific_visualizations
)
from utils import (
    load_css, validate_dataset, get_image_transform, check_image_quality,
    gradient_text, plot_cell_distribution, generate_report,
    generate_advanced_report, save_cell_specific_images
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
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

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
        <div style="font-size: 2rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center;">�</div>
        <h2 style="font-family: 'Inter', sans-serif; margin-bottom: 0.3rem; font-size: 1.8rem; font-weight: 700; color: #2d3748;">
            Blood Cell Analysis
        </h2>
        <p style="font-size: 1rem; margin: 0; color: #4a5568; font-weight: 500;">
            AI-Powered Analysis • Detailed Cell Detection • Advanced Reporting
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Image input and analysis section
st.markdown("""
<div class="upload-section" style="max-width: 800px; margin: 2rem auto; padding: 2rem; background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); border-radius: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 2rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center;">�</div>
        <h3 style="color: #2d3748; margin: 0 0 0.5rem 0; font-weight: 700; font-family: 'Inter', sans-serif;">Upload Blood Smear Image</h3>
        <p style="color: #4a5568; margin: 0; font-size: 0.9rem;">Select a blood smear image for detailed cell analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)
    
# Main file uploader section with improved styling
st.markdown("""
<div class="uploader-container" style="margin: 2rem auto; text-align: center;">
    <div style="max-width: 600px; margin: 0 auto; padding: 2rem; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="font-size: 1.5rem; margin-bottom: 1rem;">🔬</div>
        <h3 style="color: #2d3748; margin-bottom: 1rem; font-weight: 600;">Blood Cell Analysis</h3>
        <p style="color: #4a5568; margin-bottom: 1.5rem;">Upload a blood smear image for detailed analysis</p>
        <div style="font-size: 0.875rem; color: #718096;">Supported formats: JPG, JPEG, PNG (Max 200MB)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

def process_blood_cell_image(image):
    """Process blood cell image and return detection results."""
    try:
        with st.spinner("Analyzing blood cells..."):
            # Perform enhanced detection
            detections = enhance_cell_detection(image)
            if not detections:
                st.error("Could not detect cells in the image. Please try a different image.")
                return None
            
            # Generate cell-specific visualizations
            visualizations = create_cell_specific_visualizations(image, detections)
            if not visualizations:
                st.error("Could not generate cell visualizations.")
                return None
            
            # Save visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_paths = save_cell_specific_images(visualizations, f"detection_{timestamp}")
            
            # Display results
            st.markdown("### Cell Type Detection Results")
            # Create three columns for different cell type visualizations
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            # Display cell-specific images
            with viz_col1:
                st.markdown("#### RBC Detection")
                st.image(visualizations.get('RBC_visualization', ''), 
                        caption=f"RBCs Detected: {detections['stats'].get('RBC_count', 0)}", 
                        use_column_width=True)
            
            with viz_col2:
                st.markdown("#### WBC Detection")
                st.image(visualizations.get('WBC_visualization', ''), 
                        caption=f"WBCs Detected: {detections['stats'].get('WBC_count', 0)}", 
                        use_column_width=True)
            
            with viz_col3:
                st.markdown("#### Platelet Detection")
                st.image(visualizations.get('Platelet_visualization', ''), 
                        caption=f"Platelets Detected: {detections['stats'].get('Platelet_count', 0)}", 
                        use_column_width=True)
            
            return detections, saved_paths
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Process uploaded file
if uploaded_file is not None:
    try:
        # Display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        st.session_state.current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if st.session_state.current_image is None or not isinstance(st.session_state.current_image, np.ndarray):
            st.error("Could not read the uploaded image. Please try a different file.")
            st.session_state.current_image = None
        else:
            # Show the original image
            st.image(st.session_state.current_image, caption="Uploaded Blood Smear Image", use_column_width=True)
            
            # Automatically start analysis
            # Process the image and generate report
            results = process_blood_cell_image(st.session_state.current_image)
            if not results:
                st.error("Could not process the image. Please try a different image.")
            else:
                detections, saved_paths = results
                if detections and saved_paths:
                    # Generate and display report
                    report_data = generate_advanced_report(detections, uploaded_file.name)
                    if report_data:
                        # Display report metrics
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            if 'visualization_paths' in report_data and 'distribution_pie' in report_data['visualization_paths']:
                                st.image(report_data['visualization_paths']['distribution_pie'], 
                                       caption="Cell Type Distribution", 
                                       use_column_width=True)
                        
                        with metrics_col2:
                            if 'visualization_paths' in report_data and 'confidence_radar' in report_data['visualization_paths']:
                                st.image(report_data['visualization_paths']['confidence_radar'], 
                                       caption="Detection Confidence by Cell Type", 
                                       use_column_width=True)
                        
                        if 'visualization_paths' in report_data and 'metrics_bar' in report_data['visualization_paths']:
                            st.markdown("#### Detailed Cell Counts")
                            st.image(report_data['visualization_paths']['metrics_bar'], 
                                   caption="Cell Counts by Type", 
                                   use_column_width=True)
                        
                        # Add download section
                        st.markdown("### Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'report_path' in report_data:
                                with open(report_data['report_path'], 'rb') as f:
                                    st.download_button(
                                        label="📄 Download Detailed Report",
                                        data=f,
                                        file_name="blood_cell_analysis_report.txt",
                                        mime="text/plain"
                                    )
                        
                        with col2:
                            for viz_type, viz_path in saved_paths.items():
                                if os.path.exists(viz_path):
                                    with open(viz_path, 'rb') as f:
                                        st.download_button(
                                            label=f"📊 Download {viz_type.replace('_', ' ')}",
                                            data=f,
                                            file_name=f"{viz_type}_{uploaded_file.name}",
                                            mime="image/png"
                                        )
                    else:
                        st.warning("Could not generate the analysis report. The detection results are still valid.")
                else:
                    st.error("Error generating visualizations")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        if debug_mode:
            st.exception(e)



# Analysis options
st.markdown("""
<div class="glass-effect" style="padding: 1.5rem;">
    <h3 style="color: #2d3748; margin-bottom: 1rem; text-align: center; font-size: 1.4rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
        <span style="font-size: 1.8rem;">📋</span>
        <span>Select Analysis Type</span>
    </h3>
    <p style="color: #4a5568; text-align: center; margin-bottom: 1.5rem; font-size: 1rem; font-weight: 500;">Choose one or more analysis types, then click "Analyze" to proceed.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    cell_detection = st.checkbox(
        "Blood Cell Detection",
        value=True,  # Default to selected
        help="Detect and count all blood cells (RBC, WBC, Platelets).",
        key="detection_checkbox"
    )

with col2:
    morphology_analysis = st.checkbox(
        "Cell Morphology Analysis",
        value=True,  # Default to selected
        help="Analyze cell shape, size, and other characteristics.",
        key="morphology_checkbox"
    )

with col3:
    complete_analysis = st.checkbox(
        "Complete Blood Analysis",
        value=True,  # Default to selected
        help="Perform a full blood cell detection and morphology assessment.",
        key="complete_checkbox"
    )

# Analyze button
if st.button("Start Comprehensive Analysis", type="primary", key="analyze_button"):
    if st.session_state.current_image is None or not isinstance(st.session_state.current_image, np.ndarray) or st.session_state.current_image.size == 0:
        st.warning("Please upload a blood smear image before starting the analysis.")
        st.stop()
    
    selected_analyses = []
    if cell_detection:
        selected_analyses.append("detection")
    if morphology_analysis:
        selected_analyses.append("morphology")
    if complete_analysis:
        selected_analyses.append("complete")
    
    if not selected_analyses:
        st.warning("Please select at least one type of analysis.")
        st.stop()
    
    # Run blood cell analysis with improved error handling
    analysis_container = st.empty()
    analysis_container.info("Starting comprehensive blood smear analysis...")
    
    try:
        # Check image quality before proceeding
        quality_score = check_image_quality(st.session_state.current_image)
        
        # Describe the image using the BLIP model
        from utils import describe_image
        image_description = describe_image(st.session_state.current_image) if processor and blip_model else "A blood smear image ready for analysis."
        
        # Use the YOLO model for comprehensive blood cell detection
        if yolo_model is not None:
            # Save the image to a temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, st.session_state.current_image)
                
                # Run comprehensive cell detection to find all cells
                progress_container = st.empty()
                progress_container.info("🔍 Running comprehensive blood cell detection with maximum sensitivity...")
                
                try:
                    # Use a very low confidence threshold to maximize initial YOLO detections
                    detection_results = detect_all_cells_comprehensive(yolo_model, tmp_file.name, confidence_threshold=0.01)
                    progress_container.empty()  # Clear the progress message
                except Exception as detection_error:
                    progress_container.error(f"An error occurred during detection: {str(detection_error)}")
                    detection_results = None
                
                # Clean up the temporary file
                os.unlink(tmp_file.name)
            
            if detection_results:
                stats = detection_results.get('stats', {})
                detections = detection_results.get('detections', {})
                
                # Display a summary of the detection results
                st.success(f"Detection complete: {detection_results.get('detection_summary', 'Analysis completed.')}")
                
                # Show detailed detection statistics
                total_detected = stats.get('total_cells_detected', 0)
                if total_detected > 0:
                    st.info(f"""
                    **Comprehensive Detection Results:**
                    - **Total Cells Found:** {total_detected}
                    - **RBC:** {stats.get('RBC_count', 0)} ({stats.get('cell_distribution', {}).get('RBC_percentage', 0):.1f}%)
                    - **WBC:** {stats.get('WBC_count', 0)} ({stats.get('cell_distribution', {}).get('WBC_percentage', 0):.1f}%)
                    - **Platelets:** {stats.get('Platelet_count', 0)} ({stats.get('cell_distribution', {}).get('Platelet_percentage', 0):.1f}%)
                    - **Overall Confidence:** {stats.get('confidence_scores', {}).get('Overall', 0):.2%}
                    - **Detection Density:** {stats.get('detection_density', 0):.6f} cells/pixel
                    """)
                    
                    # Create and display visualizations of all detected cells
                    viz_container = st.empty()
                    viz_container.info("Generating comprehensive cell visualizations...")
                    
                    try:
                        # Save the current image to a temporary file for visualization
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as viz_tmp_file:
                            cv2.imwrite(viz_tmp_file.name, st.session_state.current_image)
                            
                            # Generate visualizations for each cell type
                            viz_rbc = create_cell_specific_visualizations(cv2.imread(viz_tmp_file.name), detection_results)['RBC_visualization']
                            viz_wbc = create_cell_specific_visualizations(cv2.imread(viz_tmp_file.name), detection_results)['WBC_visualization']
                            viz_platelets = create_cell_specific_visualizations(cv2.imread(viz_tmp_file.name), detection_results)['Platelet_visualization']
                            
                            # Save visualizations
                            cv2.imwrite('viz_all_cells.png', viz_rbc)
                            cv2.imwrite('viz_wbc_focus.png', viz_wbc)
                            cv2.imwrite('viz_platelets_focus.png', viz_platelets)
                            
                            viz_path_all = 'viz_all_cells.png'
                            viz_path_wbc = 'viz_wbc_focus.png'
                            viz_path_platelets = 'viz_platelets_focus.png'
                            
                            viz_container.empty()  # Clear the progress message
                            
                            # Display the three visualizations in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if viz_path_all and os.path.exists(viz_path_all):
                                    st.image(viz_path_all, caption=f"RBC Detection ({stats.get('RBC_count', 0)} found)", use_column_width=True)
                            with col2:
                                if viz_path_wbc and os.path.exists(viz_path_wbc):
                                    st.image(viz_path_wbc, caption=f"WBC Detection ({stats.get('WBC_count', 0)} found)", use_column_width=True)
                            with col3:
                                if viz_path_platelets and os.path.exists(viz_path_platelets):
                                    st.image(viz_path_platelets, caption=f"Platelet Detection ({stats.get('Platelet_count', 0)} found)", use_column_width=True)
                            
                            st.success("✅ All cell visualizations generated successfully!")
                            
                            # Clean up the temporary file
                            os.unlink(viz_tmp_file.name)
                            
                    except Exception as viz_error:
                        viz_container.empty()
                        st.warning(f"⚠️ Visualization generation failed: {str(viz_error)}")
                        st.info("Detection completed, but visualization failed to generate.")
                    
                    # Generate and store the full report
                    report_text = generate_report(detection_results)
                    st.session_state.report_data = {
                        "analysis_type": "blood_cell_detection",
                        "report": report_text,
                        "image": st.session_state.current_image,
                        "image_description": image_description,
                        "quality_score": quality_score,
                        "detection_results": detection_results,
                        "cell_counts": {
                            'RBC': stats.get('RBC_count', 0),
                            'WBC': stats.get('WBC_count', 0),
                            'Platelets': stats.get('Platelet_count', 0)
                        },
                        "confidences": [stats.get('confidence_scores', {}).get('Overall', 0)]
                    }
                    
                else:
                    st.warning("⚠️ No cells were detected in the uploaded image.")
            else:
                st.error("❌ Blood cell detection failed to produce results.")
        else:
            st.error("❌ The YOLO model is not available for detection.")
            
    except Exception as e:
        analysis_container.empty()  # Clear the progress message
        st.error(f"❌ An error occurred during the analysis: {str(e)}")
        if debug_mode:
            st.exception(e)

# Display results
if 'report_data' in st.session_state and st.session_state.report_data is not None and isinstance(st.session_state.report_data, dict):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); border: none;">
        <h2 style="font-family: 'Poppins', sans-serif; color: #ffffff; margin-bottom: 0.3rem; text-align: center; font-size: 1.8rem;">Analysis Complete!</h2>
        <p style="color: rgba(255, 255, 255, 0.9); text-align: center; font-size: 1rem; margin-bottom: 0;">Your comprehensive blood cell analysis is ready.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add custom CSS for better tab visibility
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 10px 16px;
            background-color: white;
            border-radius: 4px;
            margin-right: 4px;
            border: 1px solid #dee2e6;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for results with clear sections
    tab_overview, tab_analysis, tab_clinical, tab_report = st.tabs([
        "📊 Detection Results", 
        "🔬 Analysis Details", 
        "👨‍⚕️ Clinical Report",
        "📈 Full Report"
    ])
    
    # Detection Results Tab
    with tab_overview:
        st.markdown("### 📊 Detection Results")
        
        # Image and Summary in columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if st.session_state.report_data and "image" in st.session_state.report_data:
                st.image(st.session_state.report_data["image"], caption="Analyzed Blood Smear Image", use_column_width=True)
            else:
                st.info("No image available.")
                
        with col2:
            st.markdown("#### Detection Summary")
            if "cell_counts" in st.session_state.report_data:
                counts = st.session_state.report_data["cell_counts"]
                
                # Create metrics with color coding
                st.metric("Total RBCs", f"{counts.get('RBC', 0):,}")
                st.metric("Total WBCs", f"{counts.get('WBC', 0):,}")
                st.metric("Total Platelets", f"{counts.get('Platelets', 0):,}")
                
                # Add confidence score
                if "confidences" in st.session_state.report_data:
                    conf = st.session_state.report_data["confidences"]
                    if conf:
                        avg_conf = sum(conf) / len(conf)
                        st.progress(avg_conf, text=f"Overall Detection Confidence: {avg_conf:.1%}")
            else:
                st.info("No detection data available.")
                
    # Detailed Analysis Tab
    with tab_analysis:
        st.markdown("### 🔬 Detailed Analysis")
        if "report_data" in st.session_state and "cell_counts" in st.session_state.report_data:
            # Cell Distribution Analysis
            st.markdown("#### Cell Distribution Analysis")
            counts = st.session_state.report_data["cell_counts"]
            total = sum(counts.values())
            
            if total > 0:
                # Calculate percentages
                rbc_pct = (counts.get('RBC', 0) / total) * 100
                wbc_pct = (counts.get('WBC', 0) / total) * 100
                plt_pct = (counts.get('Platelets', 0) / total) * 100
                
                # Display distribution chart
                dist_data = pd.DataFrame({
                    'Cell Type': ['RBCs', 'WBCs', 'Platelets'],
                    'Percentage': [rbc_pct, wbc_pct, plt_pct]
                })
                st.bar_chart(dist_data, x='Cell Type', y='Percentage')
                
                # Add morphology analysis if available
                if "image_description" in st.session_state.report_data:
                    st.markdown("#### Morphological Analysis")
                    st.write(st.session_state.report_data["image_description"])
                    
                # Add quality metrics
                if "quality_score" in st.session_state.report_data:
                    st.markdown("#### Quality Metrics")
                    quality = st.session_state.report_data["quality_score"]
                    st.progress(quality, text=f"Image Quality Score: {quality:.1%}")
            else:
                st.warning("No cells were detected for analysis.")
            
    # Clinical Report Tab
    with tab_clinical:
        st.markdown("### 👨‍⚕️ Clinical Report")
        if 'report_data' in st.session_state and st.session_state.report_data.get('cell_counts'):
            counts = st.session_state.report_data['cell_counts']
            
            # Display clinical analysis from the generated report
            if "report" in st.session_state.report_data:
                st.markdown(st.session_state.report_data["report"])
            else:
                st.warning("No clinical report was generated.")
    
    # Full Report Tab
    with tab_report:
        st.markdown("### 📈 Full Analysis Report")
        if 'report_data' in st.session_state and "report" in st.session_state.report_data:
            st.markdown(st.session_state.report_data["report"])
            
            # Download report button
            st.download_button(
                label="Download Full Report",
                data=st.session_state.report_data["report"],
                file_name="blood_cell_analysis_report.txt",
                mime="text/plain"
            )
        else:
            st.warning("No report is available to display or download.")
else:
    st.info("Upload an image and start the analysis to see the results.")

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