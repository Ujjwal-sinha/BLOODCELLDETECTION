"""
Streamlit App for Blood Cell Detection - BloodCellAI
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO  # Added to fix NameError
from models import load_yolo_model, predict_blood_cells
from utils import (
    load_css, check_image_quality, describe_image, create_lime_visualization,
    create_shap_visualization, create_gradcam_visualization, query_langchain, BloodPDF, gradient_text
)
from agents import BloodCellAgent
import base64

# Set page configuration
st.set_page_config(page_title="BloodCellAI - Blood Cell Detection", layout="wide")

# Load custom CSS
load_css()

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None

# Load YOLO model
model_path = "/Users/ujjwalsinha/Blood Cell Detection/yolo11n.pt"
classes = ['Platelets', 'RBC', 'WBC']
model = load_yolo_model(model_path, num_classes=len(classes))

# Function to convert image to base64 for HTML rendering
def image_to_base64(image):
    """Convert a PIL Image or numpy array to base64 string for HTML embedding"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Main app
def main():
    st.markdown(
        f"""
        <div class="gradient-header">
            <h1>{gradient_text("BloodCellAI - AI-Powered Blood Cell Detection", "#667eea", "#764ba2")}</h1>
            <h2 class="subtitle">Upload a blood cell image to detect Platelets, RBC, and WBC</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for model controls
    with st.sidebar:
        st.markdown('<h3 data-icon="🛠">Model Controls</h3>', unsafe_allow_html=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, key="conf_slider")
        retrain = st.button("Retrain Model", key="retrain_btn", help="Retrain the YOLO model", type="primary")
        if retrain:
            st.markdown('<div class="info-box">Model retraining not implemented in this version.</div>', unsafe_allow_html=True)
            # Add retraining logic here if needed

    # File uploader
    st.markdown('<div class="glass-effect"><h3>Upload Blood Cell Image</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'], key="file_uploader")

    if uploaded_file:
        # Save and process uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.uploaded_image = image

        # Check image quality
        quality_score = check_image_quality(image)
        if quality_score < 0.5:
            st.markdown(
                f'<div class="error-box">Low image quality (Score: {quality_score:.2f}). Consider uploading a clearer image.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="success-box">Image quality acceptable (Score: {quality_score:.2f}).</div>',
                unsafe_allow_html=True
            )

        # Display uploaded image
        st.markdown('<div class="glass-effect"><h3>Uploaded Image</h3></div>', unsafe_allow_html=True)
        img_base64 = image_to_base64(image)
        st.markdown(
            f'<div class="zoomable-image"><img src="data:image/png;base64,{img_base64}" alt="Uploaded Blood Cell Image" style="width:100%;"></div>',
            unsafe_allow_html=True
        )
        st.caption("Uploaded Blood Cell Image")

        # Perform detection
        with st.spinner("Detecting blood cells..."):
            predictions = predict_blood_cells(model, image, classes, confidence_threshold=confidence_threshold)
            st.session_state.predictions = predictions

        # Display detection results
        if predictions and predictions['detections']:
            st.markdown('<div class="analysis-container"><h3>Detection Results</h3></div>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Draw bounding boxes
                img_array = np.array(image)
                for det in predictions['detections']:
                    box = det['box']
                    label = det['class']
                    conf = det['confidence']
                    x1, y1, x2, y2 = map(int, box)
                    color = (255, 0, 0) if label == 'RBC' else (0, 255, 0) if label == 'WBC' else (0, 0, 255)
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_array, f"{label} {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # Display image with bounding boxes
                img_base64 = image_to_base64(img_array)
                st.markdown(
                    f'<div class="zoomable-image"><img src="data:image/png;base64,{img_base64}" alt="Detected Blood Cells" style="width:100%;"></div>',
                    unsafe_allow_html=True
                )
                st.caption("Detected Blood Cells")

            with col2:
                st.markdown('<div class="compact-card"><h4>Detected Cells</h4></div>', unsafe_allow_html=True)
                cell_counts = {'Platelets': 0, 'RBC': 0, 'WBC': 0}
                for det in predictions['detections']:
                    cell_counts[det['class']] += 1
                for cell_type, count in cell_counts.items():
                    st.markdown(f'<div class="compact-card"><p>{cell_type}: {count}</p></div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="error-box">No cells detected. Try adjusting the confidence threshold or uploading a different image.</div>', unsafe_allow_html=True)

        # AI Analysis
        if st.button("Generate AI Analysis", key="analysis_btn", help="Generate detailed AI analysis", type="primary"):
            with st.spinner("Generating AI analysis..."):
                image_description = describe_image(image)
                ai_agent = BloodCellAgent()
                analysis = ai_agent.analyze_blood_cells(image_description, predictions)
                st.session_state.ai_analysis = analysis
                st.markdown('<div class="analysis-container"><h3>AI Analysis</h3></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="analysis-card">{analysis}</div>', unsafe_allow_html=True)

        # Generate PDF Report
        if st.button("Generate PDF Report", key="download_btn", help="Download analysis report", type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf = BloodPDF(cell_info="Blood Cell Detection Report")
                pdf.cover_page()
                pdf.table_of_contents()
                
                # Add analysis sections
                image_description = describe_image(image)
                pdf.add_summary(st.session_state.ai_analysis or "No AI analysis available", 
                               cell_context=image_description)
                
                # Save uploaded image temporarily
                temp_image_path = "temp_image.jpg"
                image.save(temp_image_path)
                pdf.add_image(temp_image_path)
                
                # Add detection results
                if st.session_state.predictions and st.session_state.predictions['detections']:
                    detection_summary = "\n".join([f"{det['class']}: {det['confidence']:.2f}" 
                                                 for det in st.session_state.predictions['detections']])
                    pdf.add_section("Detection Results", detection_summary)
                
                # Add visualizations
                predicted_class = predictions['detections'][0]['class'] if predictions and predictions['detections'] else "Unknown"
                lime_path = create_lime_visualization(image, model, classes)
                shap_path = create_shap_visualization(image, model, classes)
                gradcam_path = create_gradcam_visualization(image, model, classes)
                pdf.add_explainability(lime_path, None, shap_path)
                
                # Save PDF
                pdf_path = "blood_cell_report.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="blood_cell_report.pdf",
                        mime="application/pdf",
                        key="download_pdf_btn",
                        help="Download the generated report",
                        type="primary"
                    )
                # Clean up temporary files
                for path in [temp_image_path, lime_path, shap_path, gradcam_path]:
                    if path and os.path.exists(path):
                        os.remove(path)

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>BloodCellAI &copy; 2025 | Powered by xAI</p>
            <p><a href="https://x.ai">Visit xAI</a> | <a href="https://github.com">Source Code</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()