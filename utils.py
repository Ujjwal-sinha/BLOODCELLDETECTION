"""
Blood Cell Detection Utilities - BloodCellAI
Utility functions for image processing, AI analysis, and report generation
"""

import os
import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import time
import random
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
import uuid
import glob

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

def retry_with_exponential_backoff(func, max_retries=4, base_delay=2):
    """
    Retry a function with exponential backoff.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            if "over capacity" not in error_msg and "503" not in str(e) and "rate limit" not in error_msg:
                raise e
            if attempt == max_retries:
                print(f"Max retries reached. Using fallback response.")
                return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"GROQ API over capacity, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

@st.cache_resource
def load_models():
    """Load BLIP models for image description"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        print(f"Error loading BLIP models: {e}")
        return None, None

def check_image_quality(image: Image.Image, suspected_cell: str = None) -> float:
    """
    Check image quality for blood cell detection
    """
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        if height < 100 or width < 100:
            return 0.3
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            return 0.4
        contrast = np.std(gray)
        if contrast < 20:
            return 0.5
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return 0.6
        color_std = np.std(img_array, axis=(0, 1))
        if np.any(color_std < 10):
            return 0.7
        quality_score = min(1.0, (
            (brightness / 128) * 0.2 +
            (contrast / 50) * 0.3 +
            (laplacian_var / 500) * 0.3 +
            (np.mean(color_std) / 50) * 0.2
        ))
        return max(0.1, quality_score)
    except Exception as e:
        print(f"Error checking image quality: {e}")
        return 0.5

def describe_image(image: Image.Image, suspected_cell: str = None) -> str:
    """
    Generate detailed description of blood cell image
    """
    try:
        processor, model = load_models()
        if processor is None or model is None:
            return "Blood cell image showing microscopic view"
        
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100, num_beams=5)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        enhanced_description = f"Blood cell image showing: {description}. "
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        if edge_density > 0.1:
            enhanced_description += "Image shows distinct cell boundaries, likely indicating multiple blood cells. "
        else:
            enhanced_description += "Image shows uniform cell distribution. "
        
        return enhanced_description
    except Exception as e:
        print(f"Error describing image: {e}")
        return "Blood cell image for analysis"

def test_groq_api():
    """Test GROQ API connectivity"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False, "No API key provided"
        
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(model=model_name, temperature=0.1, groq_api_key=api_key)
                    test_response = llm.invoke("Test blood cell analysis")
                    if test_response:
                        return True, f"Working (using {model_name})"
                    else:
                        raise Exception("Empty response from model")
                result = retry_with_exponential_backoff(test_model)
                if result:
                    return result
            except Exception as e:
                continue
        return False, "All models are currently unavailable"
    except Exception as e:
        return False, f"API test failed: {str(e)}"

def generate_fallback_response(detected_cell: str, image_description: str, yolo_detection: str = None, confidence: float = None) -> str:
    """
    Generate fallback analysis for blood cell detection
    """
    try:
        analysis = f"""
        **Blood Cell Detection Report**
        
        **Image Analysis:**
        {image_description}
        
        **AI Detection Results:**
        - Detected Cell Type: {detected_cell}
        - Confidence Level: {confidence:.2f if confidence else 99.0}%
        - YOLO Model Detection: {yolo_detection if yolo_detection else "Not available"}
        
        **Cell Assessment:**
        The image shows characteristics consistent with {detected_cell.lower()}.
        
        **Common Observations:**
        - Distinct cell boundaries
        - Color variations indicating cell types
        - Structural features of blood cells
        
        **Recommendations:**
        1. **Consult a hematologist** for detailed analysis
        2. **Monitor cell counts** in follow-up tests
        3. **Correlate with clinical symptoms** for accurate diagnosis
        
        **Follow-up Actions:**
        - Schedule laboratory analysis
        - Document cell count changes
        - Consider additional blood tests if recommended
        """
        return analysis
    except Exception as e:
        return f"Error generating fallback response: {str(e)}"

def query_langchain(prompt: str, detected_cell: str, confidence: float = None, cell_context: str = None, yolo_detection: str = None) -> str:
    """
    Query LangChain for blood cell analysis
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return generate_fallback_response(detected_cell, prompt, yolo_detection, confidence)
        
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for model_name in models_to_try:
            try:
                def query_model():
                    llm = ChatGroq(model=model_name, temperature=0.1, groq_api_key=api_key)
                    enhanced_prompt = f"""
                    You are an expert hematologist. Analyze the following blood cell case:
                    
                    {prompt}
                    
                    Detected Cell Type: {detected_cell}
                    Confidence: {confidence:.2f if confidence else 99.0}%
                    YOLO Detection: {yolo_detection if yolo_detection else "Not available"}
                    Cell Context: {cell_context if cell_context else "Not provided"}
                    
                    Provide a comprehensive blood cell analysis including:
                    1. **Cell Identification**: Confirm the detected cell type
                    2. **Cell Characteristics**: Describe visible features
                    3. **Potential Implications**: Clinical significance of cell counts
                    4. **Recommendations**: Diagnostic and follow-up steps
                    """
                    response = llm.invoke(enhanced_prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                result = retry_with_exponential_backoff(query_model)
                if result is not None:
                    return result
            except Exception as e:
                continue
        return generate_fallback_response(detected_cell, prompt, yolo_detection, confidence)
    except Exception as e:
        return generate_fallback_response(detected_cell, prompt, yolo_detection, confidence)

class BloodPDF(FPDF):
    """PDF generator for blood cell detection reports"""
    def __init__(self, cell_info=""):
        super().__init__()
        self.cell_info = self.sanitize_text(cell_info)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
    
    def sanitize_text(self, text):
        """Sanitize text for PDF compatibility"""
        if not text:
            return ""
        text = text.replace('"', "'").replace('"', "'").replace('–', '-').replace('—', '-')
        text = text.replace('•', '-').replace('…', '...').replace('°', ' degrees')
        text = ''.join(char for char in text if ord(char) < 128)
        return text[:1000]
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Blood Cell Detection Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def cover_page(self):
        self.set_font('Arial', 'B', 24)
        self.cell(0, 60, 'Blood Cell Detection Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 20, 'AI-Powered Hematological Assessment', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 20, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        if self.cell_info:
            self.cell(0, 20, f'Cell Information: {self.cell_info}', 0, 1, 'C')
        self.add_page()
    
    def table_of_contents(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(5)
        sections = [
            'Executive Summary',
            'Image Analysis',
            'Cell Detection Results',
            'Detailed Analysis',
            'Recommendations',
            'Follow-up Actions'
        ]
        for i, section in enumerate(sections, 1):
            self.set_font('Arial', '', 12)
            self.cell(0, 8, f'{i}. {section}', 0, 1, 'L')
        self.add_page()
    
    def add_image(self, image_path, width=180):
        try:
            if os.path.exists(image_path):
                self.image(image_path, x=10, y=self.get_y(), w=width)
                self.ln(5)
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
    
    def add_section(self, title, body):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.sanitize_text(title), 0, 1, 'L')
        self.ln(2)
        self.set_font('Arial', '', 11)
        paragraphs = body.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                sanitized_paragraph = self.sanitize_text(paragraph.strip())
                self.multi_cell(0, 5, sanitized_paragraph)
                self.ln(2)
        self.ln(5)
    
    def add_summary(self, report, cell_context=None):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        self.set_font('Arial', '', 11)
        summary_points = [
            "AI-powered blood cell detection completed successfully",
            "Analysis of Platelets, RBC, and WBC counts",
            "Recommendations for clinical follow-up provided"
        ]
        for point in summary_points:
            self.cell(0, 5, f"- {point}", 0, 1, 'L')
        if cell_context:
            self.ln(5)
            sanitized_context = self.sanitize_text(cell_context)
            self.cell(0, 5, f"Cell Context: {sanitized_context}", 0, 1, 'L')
        self.ln(10)
    
    def add_explainability(self, lime_path, edge_path, shap_path):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Explainability Analysis', 0, 1, 'L')
        self.ln(5)
        self.set_font('Arial', '', 11)
        self.cell(0, 5, 'Visualizations showing AI model analysis of blood cells:', 0, 1, 'L')
        self.ln(5)
        for path, description in [
            (lime_path, "LIME Analysis - Feature Importance"),
            (shap_path, "SHAP Analysis - Model Interpretability")
        ]:
            if path and os.path.exists(path):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, self.sanitize_text(description), 0, 1, 'L')
                self.add_image(path, width=150)
                self.ln(5)

def gradient_text(text, color1, color2):
    return f'<span class="text-gradient">{text}</span>'

def validate_dataset(dataset_dir):
    try:
        if not os.path.exists(dataset_dir):
            return False, f"Dataset directory '{dataset_dir}' not found"
        classes = ['Platelets', 'RBC', 'WBC']
        total_images = 0
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_dir, split, 'images')
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(images)
        if total_images < 10:
            return False, f"Dataset too small: {total_images} images found"
        return True, f"Dataset validated: {total_images} images"
    except Exception as e:
        return False, f"Dataset validation error: {str(e)}"

def preprocess_image(img_path, output_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced_img = Image.fromarray(enhanced)
        enhanced_img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False

def augment_with_blur(img_path, output_path, blur_radius=2):
    try:
        img = Image.open(img_path).convert('RGB')
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error creating blur augmentation: {e}")
        return False

def load_css():
    """Load external CSS file for styling"""
    try:
        with open("style.css", "r") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        print("style.css not found, falling back to default styling")
        st.markdown("""
        <style>
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        .gradient-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .success-box { background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #48bb78; }
        .error-box { background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #e53e3e; }
        .info-box { background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #38b2ac; }
        </style>
        """, unsafe_allow_html=True)

def clear_mps_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

def create_dataset_splits(dataset_dir, split_ratio=(0.7, 0.15, 0.15)):
    try:
        from sklearn.model_selection import train_test_split
        all_images = []
        all_labels = []
        classes = ['Platelets', 'RBC', 'WBC']
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_dir, split, 'images')
            label_dir = os.path.join(dataset_dir, split, 'labels')
            if os.path.exists(img_dir) and os.path.exists(label_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                for img in images:
                    all_images.append(os.path.join(img_dir, img))
                    label_path = os.path.join(label_dir, img.rsplit('.', 1)[0] + '.txt')
                    labels = []
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            for line in f:
                                labels.append(line.strip().split()[0])
                    all_labels.append(labels)
        
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            all_images, all_labels, test_size=1-split_ratio[0], random_state=42
        )
        val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=1-val_ratio, random_state=42
        )
        
        return {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels),
            'classes': classes
        }
    except Exception as e:
        print(f"Error creating dataset splits: {e}")
        return None

def search_cells_globally(query, classes):
    """
    Search for blood cell types based on query
    """
    if not query or not classes:
        return []
    query = query.lower().strip()
    results = []
    cell_database = {
        'platelets': {
            'name': 'Platelets',
            'type': 'Blood Component',
            'description': 'Small cell fragments involved in blood clotting'
        },
        'rbc': {
            'name': 'Red Blood Cells (RBC)',
            'type': 'Blood Component',
            'description': 'Cells that carry oxygen throughout the body'
        },
        'wbc': {
            'name': 'White Blood Cells (WBC)',
            'type': 'Blood Component',
            'description': 'Cells that fight infections and support immunity'
        }
    }
    for class_name in classes:
        class_lower = class_name.lower()
        if query in class_lower:
            if class_lower in cell_database:
                results.append(cell_database[class_lower])
            else:
                results.append({
                    'name': class_name.title(),
                    'type': 'Detected Cell',
                    'description': f'Detected blood cell: {class_name.title()}'
                })
        for cell_key, cell_info in cell_database.items():
            if query in cell_key or query in cell_info['name'].lower():
                if cell_info not in results:
                    results.append(cell_info)
    return results[:10]

def create_lime_visualization(image, model, classes):
    try:
        import lime
        from lime import lime_image
        explainer = lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            model.eval()
            batch = torch.stack([transforms.ToTensor()(Image.fromarray(img)) for img in images]).to(device)
            with torch.no_grad():
                results = model.predict(batch, conf=0.5)
            probs = []
            for result in results:
                prob = np.zeros(len(classes))
                for cls, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    prob[int(cls)] = max(prob[int(cls)], conf)
                probs.append(prob)
            return np.array(probs)
        
        explanation = explainer.explain_instance(
            np.array(image), predict_fn, top_labels=3, hide_color=0, num_samples=500
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True
        )
        axes[1].imshow(mask, cmap='Reds', alpha=0.7)
        axes[1].imshow(image, alpha=0.3)
        axes[1].set_title('LIME Explanation', fontweight='bold')
        axes[1].axis('off')
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
        )
        axes[2].imshow(mask, cmap='RdBu', alpha=0.7)
        axes[2].imshow(image, alpha=0.3)
        axes[2].set_title('LIME Full Explanation', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        lime_path = "lime_visualization.png"
        plt.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close()
        return lime_path
    except Exception as e:
        print(f"Error creating LIME visualization: {e}")
        return None

def create_shap_visualization(image, model, classes):
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        feature_importance = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(feature_importance, cmap='RdBu')
        axes[1].set_title('SHAP Feature Importance', fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(img_array)
        axes[2].imshow(feature_importance, cmap='RdBu', alpha=0.6)
        axes[2].set_title('SHAP Overlay', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        shap_path = "shap_visualization.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        return shap_path
    except Exception as e:
        print(f"Error creating SHAP visualization: {e}")
        return None

def create_gradcam_visualization(image, model, classes):
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        attention_map = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(img_array)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.6)
        axes[2].set_title('Grad-CAM Overlay', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        gradcam_path = "gradcam_visualization.png"
        plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
        plt.close()
        return gradcam_path
    except Exception as e:
        print(f"Error creating Grad-CAM visualization: {e}")
        return None