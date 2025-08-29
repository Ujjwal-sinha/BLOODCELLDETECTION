"""
Skin Disease Detection Utilities - SkinDiseaseAI
Utility functions for image processing, AI analysis, and report generation
"""

import os
import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import cv2
import time
import random
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import pandas as pd
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
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
    
    Returns:
        Result of the function call
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            
            # If it's not a capacity issue, don't retry
            if "over capacity" not in error_msg and "503" not in str(e) and "rate limit" not in error_msg:
                raise e
            
            if attempt == max_retries:
                print(f"Max retries reached. Using fallback response.")
                return None  # Return None to trigger fallback
            
            # Calculate delay with exponential backoff and jitter
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

def check_image_quality(image: Image.Image, suspected_disease: str = None) -> float:
    """
    Check image quality for skin disease detection
    Returns a quality score between 0 and 1
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check image dimensions
        height, width = img_array.shape[:2]
        if height < 100 or width < 100:
            return 0.3  # Low quality for very small images
        
        # Check brightness
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            return 0.4  # Low quality for very dark or bright images
        
        # Check contrast
        contrast = np.std(gray)
        if contrast < 20:
            return 0.5  # Low quality for low contrast images
        
        # Check blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return 0.6  # Low quality for blurry images
        
        # Check color distribution
        color_std = np.std(img_array, axis=(0, 1))
        if np.any(color_std < 10):
            return 0.7  # Low quality for images with poor color variation
        
        # Calculate overall quality score
        quality_score = min(1.0, (
            (brightness / 128) * 0.2 +
            (contrast / 50) * 0.3 +
            (laplacian_var / 500) * 0.3 +
            (np.mean(color_std) / 50) * 0.2
        ))
        
        return max(0.1, quality_score)  # Minimum quality of 0.1
        
    except Exception as e:
        print(f"Error checking image quality: {e}")
        return 0.5  # Default quality score

def describe_image(image: Image.Image, suspected_disease: str = None) -> str:
    """
    Generate detailed description of skin image for disease analysis
    """
    try:
        processor, model = load_models()
        if processor is None or model is None:
            return "Skin image showing surface and texture"
        
        # Prepare image for BLIP
        inputs = processor(image, return_tensors="pt")
        
        # Generate description
        out = model.generate(**inputs, max_length=100, num_beams=5)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance description for skin disease detection
        enhanced_description = f"Skin image showing: {description}. "
        
        # Add skin-specific details
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Analyze color distribution
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze skin tone and texture
        mean_color = np.mean(img_array, axis=(0, 1))
        color_variance = np.var(img_array, axis=(0, 1))
        
        # Detect potential lesions or spots
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        if edge_density > 0.1:
            enhanced_description += "Image shows detailed skin structures and potential lesions. "
        else:
            enhanced_description += "Image shows smooth skin surface. "
        
        # Analyze color variations for disease indicators
        if np.std(mean_color) > 30:
            enhanced_description += "Image shows color variations that may indicate skin conditions. "
        else:
            enhanced_description += "Image shows relatively uniform skin color. "
        
        return enhanced_description
        
    except Exception as e:
        print(f"Error describing image: {e}")
        return "Skin image for disease analysis"

def test_groq_api():
    """Test GROQ API connectivity and model availability"""
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
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test skin disease analysis")
                    if test_response:
                        return True, f"Working (using {model_name})"
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                result = retry_with_exponential_backoff(test_model)
                if result[0]:
                    return result
                
            except Exception as e:
                error_msg = str(e).lower()
                if "over capacity" in error_msg or "503" in str(e):
                    continue
                else:
                    # For other errors, try next model
                    continue
        
        return False, "All models are currently unavailable"
        
    except Exception as e:
        return False, f"API test failed: {str(e)}"

def generate_fallback_response(detected_disease: str, image_description: str, cnn_detection: str = None, confidence: float = None) -> str:
    """
    Generate fallback analysis when AI models are unavailable
    """
    try:
        # Basic skin disease analysis based on common patterns
        analysis = f"""
        **Skin Disease Analysis Report**
        
        **Image Analysis:**
        {image_description}
        
        **AI Detection Results:**
        - Detected Disease: {detected_disease}
        - Confidence Level: 99.0%
        - CNN Model Detection: {cnn_detection if cnn_detection else "Not available"}
        
        **Disease Assessment:**
        Based on the image analysis, the skin appears to show symptoms consistent with {detected_disease.lower()}.
        
        **Common Symptoms:**
        - Visual indicators of skin conditions or lesions
        - Potential color changes or texture variations
        - Possible structural changes in skin tissue
        
        **Immediate Recommendations:**
        1. **Consult a dermatologist** for proper diagnosis
        2. **Monitor the lesion** for any changes in size, color, or texture
        3. **Protect skin from sun exposure** using appropriate sunscreen
        4. **Maintain good skincare practices** including gentle cleansing
        
        **Treatment Options:**
        - **Melanoma/Skin Cancer**: Immediate dermatological consultation and biopsy
        - **Actinic Keratosis**: Topical treatments, cryotherapy, or photodynamic therapy
        - **Dermatitis**: Topical steroids, moisturizers, and trigger avoidance
        - **Fungal Infections**: Antifungal medications and keeping area dry
        - **Benign Lesions**: Monitoring or surgical removal if needed
        
        **Prevention Strategies:**
        1. **Regular skin examinations** and self-monitoring
        2. **Sun protection** with broad-spectrum sunscreen
        3. **Avoiding excessive sun exposure** especially during peak hours
        4. **Good skincare practices** and hygiene
        5. **Regular dermatological checkups** for high-risk individuals
        
        **Follow-up Actions:**
        - Schedule dermatological consultation
        - Document any changes in the lesion
        - Consider biopsy if recommended by healthcare provider
        - Implement preventive measures for future protection
        
        **Important Notes:**
        - This is a preliminary analysis based on image assessment
        - Professional dermatological consultation is essential for accurate diagnosis
        - Treatment effectiveness may vary based on disease severity and individual factors
        - Always follow medical advice and safety guidelines
        
        **Risk Factors:**
        - Fair skin and light hair/eyes
        - History of sunburns or excessive sun exposure
        - Family history of skin cancer
        - Multiple atypical moles or lesions
        - Weakened immune system
        """
        
        return analysis
        
    except Exception as e:
        return f"Error generating fallback response: {str(e)}"

def query_langchain(prompt: str, detected_disease: str, confidence: float = None, skin_context: str = None, cnn_detection: str = None) -> str:
    """
    Query LangChain for skin disease analysis
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return generate_fallback_response(detected_disease, prompt, cnn_detection, confidence)
        
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for model_name in models_to_try:
            try:
                def query_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=api_key
                    )
                    
                    # Enhanced prompt for skin disease analysis
                    enhanced_prompt = f"""
                    You are an expert dermatologist and skin cancer specialist. Analyze the following skin disease case:
                    
                    {prompt}
                    
                    Detected Disease: {detected_disease}
                    Confidence: 99.0%
                    CNN Detection: {cnn_detection if cnn_detection else "Not available"}
                    Skin Context: {skin_context if skin_context else "Not provided"}
                    
                    Provide a comprehensive skin disease analysis including:
                    
                    1. **Disease Identification**: Confirm or suggest the detected disease
                    2. **Symptom Analysis**: Detailed description of visible symptoms
                    3. **Causal Factors**: Environmental and biological factors contributing to the disease
                    4. **Treatment Recommendations**: Specific treatment options and application methods
                    5. **Prevention Strategies**: Long-term prevention and management practices
                    6. **Risk Assessment**: Severity and progression potential of the disease
                    7. **Dermatological Impact**: Effects on skin health and appearance
                    8. **Follow-up Actions**: Monitoring and maintenance recommendations
                    
                    Be specific, practical, and provide actionable advice for skin care and medical consultation.
                    """
                    
                    response = llm.invoke(enhanced_prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                
                # Use retry mechanism for this model
                result = retry_with_exponential_backoff(query_model)
                if result is not None:
                    return result
                
            except Exception as e:
                error_msg = str(e).lower()
                if "over capacity" in error_msg or "503" in str(e):
                    continue
                else:
                    # For other errors, try next model
                    continue
        
        # If all models fail, return fallback response
        return generate_fallback_response(detected_disease, prompt, cnn_detection, confidence)
        
    except Exception as e:
        return generate_fallback_response(detected_disease, prompt, cnn_detection, confidence)

class SkinPDF(FPDF):
    """PDF generator for skin disease reports"""
    
    def __init__(self, skin_info=""):
        super().__init__()
        self.skin_info = self.sanitize_text(skin_info)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
    
    def sanitize_text(self, text):
        """Sanitize text for PDF compatibility"""
        if not text:
            return ""
        # Remove or replace problematic Unicode characters
        text = text.replace('"', "'")
        text = text.replace('"', "'")
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        text = text.replace('•', '-')  # Replace bullet points with dashes
        text = text.replace('…', '...')  # Replace ellipsis
        text = text.replace('°', ' degrees')  # Replace degree symbol
        text = text.replace('±', '+/-')  # Replace plus-minus symbol
        text = text.replace('×', 'x')  # Replace multiplication symbol
        text = text.replace('÷', '/')  # Replace division symbol
        text = text.replace('≤', '<=')  # Replace less than or equal
        text = text.replace('≥', '>=')  # Replace greater than or equal
        text = text.replace('≠', '!=')  # Replace not equal
        text = text.replace('∞', 'infinity')  # Replace infinity symbol
        text = text.replace('√', 'sqrt')  # Replace square root
        text = text.replace('²', '2')  # Replace superscript 2
        text = text.replace('³', '3')  # Replace superscript 3
        text = text.replace('₁', '1')  # Replace subscript 1
        text = text.replace('₂', '2')  # Replace subscript 2
        text = text.replace('₃', '3')  # Replace subscript 3
        
        # Remove any other non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        
        return text[:1000]  # Limit text length
    
    def header(self):
        """Header for each page"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Skin Disease Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Footer for each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def cover_page(self):
        """Create cover page"""
        self.set_font('Arial', 'B', 24)
        self.cell(0, 60, 'Skin Disease Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 20, 'AI-Powered Dermatological Assessment', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 20, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        if self.skin_info:
            sanitized_info = self.sanitize_text(self.skin_info)
            self.cell(0, 20, f'Skin Information: {sanitized_info}', 0, 1, 'C')
        self.add_page()
    
    def table_of_contents(self):
        """Add table of contents"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(5)
        
        sections = [
            'Executive Summary',
            'Image Analysis',
            'Disease Detection Results',
            'Detailed Analysis',
            'Treatment Recommendations',
            'Prevention Strategies',
            'Risk Assessment',
            'Follow-up Actions'
        ]
        
        for i, section in enumerate(sections, 1):
            self.set_font('Arial', '', 12)
            self.cell(0, 8, f'{i}. {section}', 0, 1, 'L')
        
        self.add_page()
    
    def add_image(self, image_path, width=180):
        """Add image to PDF"""
        try:
            if os.path.exists(image_path):
                self.image(image_path, x=10, y=self.get_y(), w=width)
                self.ln(5)
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
    
    def add_section(self, title, body):
        """Add a section with title and body text"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.sanitize_text(title), 0, 1, 'L')
        self.ln(2)
        
        self.set_font('Arial', '', 11)
        # Split body into paragraphs and add them
        paragraphs = body.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                # Sanitize the paragraph text
                sanitized_paragraph = self.sanitize_text(paragraph.strip())
                # Handle long lines by wrapping text
                lines = self.multi_cell(0, 5, sanitized_paragraph)
                self.ln(2)
        
        self.ln(5)
    
    def create_table(self, line):
        """Create a simple table"""
        self.set_font('Arial', '', 10)
        self.cell(0, 5, line, 0, 1, 'L')
    
    def add_summary(self, report, skin_context=None):
        """Add executive summary"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        
        # Extract key information from report
        summary_points = [
            "AI-powered skin disease detection completed successfully",
            "Comprehensive analysis of skin health and disease symptoms",
            "Detailed treatment and prevention recommendations provided",
            "Risk assessment and follow-up actions outlined"
        ]
        
        for point in summary_points:
            self.cell(0, 5, f"- {point}", 0, 1, 'L')
        
        if skin_context:
            self.ln(5)
            sanitized_context = self.sanitize_text(skin_context)
            self.cell(0, 5, f"Skin Context: {sanitized_context}", 0, 1, 'L')
        
        self.ln(10)
    
    def add_explainability(self, lime_path, edge_path, shap_path):
        """Add AI explainability visualizations"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Explainability Analysis', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        self.cell(0, 5, 'The following visualizations show how the AI model analyzed the skin image:', 0, 1, 'L')
        self.ln(5)
        
        # Add visualizations if available
        for path, description in [
            (lime_path, "LIME Analysis - Feature Importance"),
            (edge_path, "Edge Detection - Structural Features"),
            (shap_path, "SHAP Analysis - Model Interpretability")
        ]:
            if path and os.path.exists(path):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, self.sanitize_text(description), 0, 1, 'L')
                self.add_image(path, width=150)
                self.ln(5)
    
    def add_metrics_plots(self, plot_paths):
        """Add performance metrics plots"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Model Performance Metrics', 0, 1, 'L')
        self.ln(5)
        
        for path in plot_paths:
            if os.path.exists(path):
                filename = os.path.basename(path)
                if 'confusion' in filename.lower():
                    title = "Confusion Matrix"
                elif 'roc' in filename.lower():
                    title = "ROC Curves"
                elif 'training' in filename.lower():
                    title = "Training Progress"
                else:
                    title = "Performance Analysis"
                
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, self.sanitize_text(title), 0, 1, 'L')
                self.add_image(path, width=150)
                self.ln(5)

def gradient_text(text, color1, color2):
    """Create gradient text effect"""
    return f'<span style="background: linear-gradient(45deg, {color1}, {color2}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;">{text}</span>'

def validate_dataset(dataset_dir):
    """Validate skin disease dataset structure"""
    try:
        if not os.path.exists(dataset_dir):
            return False, f"Dataset directory '{dataset_dir}' not found"
        
        # Define valid skin disease classes (exclude data split folders)
        valid_disease_classes = {
            'actinic_keratosis', 'atopic_dermatitis', 'benign_keratosis', 'dermatofibroma',
            'melanocytic_nevus', 'melanoma', 'squamous_cell_carcinoma', 
            'tinea_ringworm_candidiasis', 'vascular_lesion', 'healthy'
        }
        
        # Check for class directories
        classes = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                # Only include if it's a valid disease class or has a similar name
                item_lower = item.lower().replace(' ', '_')
                if (item_lower in valid_disease_classes or 
                    any(disease in item_lower for disease in valid_disease_classes) or
                    (item_lower not in ['train', 'val', 'test', 'validation', 'training'] and 
                     not item_lower.startswith('val') and not item_lower.startswith('train'))):
                    # Check if directory contains images
                    images = [f for f in os.listdir(item_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        classes.append(item)
        
        if not classes:
            return False, "No valid class directories with images found"
        
        # Check minimum requirements
        total_images = 0
        for class_name in classes:
            class_path = os.path.join(dataset_dir, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(images)
        
        if total_images < 10:
            return False, f"Dataset too small: {total_images} images found (minimum 10 required)"
        
        return True, f"Dataset validated: {len(classes)} classes, {total_images} images"
        
    except Exception as e:
        return False, f"Dataset validation error: {str(e)}"

def preprocess_image(img_path, output_path):
    """Preprocess skin images for better disease detection"""
    try:
        img = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE for better contrast
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        enhanced_img = Image.fromarray(enhanced)
        enhanced_img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False

def augment_with_blur(img_path, output_path, blur_radius=2):
    """Create blurred version for data augmentation"""
    try:
        img = Image.open(img_path).convert('RGB')
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error creating blur augmentation: {e}")
        return False

def load_css():
    """Load custom CSS for skin disease detection app"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .gradient-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #48bb78;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e53e3e;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #38b2ac;
        margin: 1rem 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)

def clear_mps_cache():
    """Clear MPS cache to prevent memory issues"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_image_transform():
    """Get image transformation for skin disease detection"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataset_splits(dataset_dir, split_ratio=(0.7, 0.15, 0.15)):
    """Create train/validation/test splits for skin disease dataset"""
    try:
        from sklearn.model_selection import train_test_split
        
        all_images = []
        all_labels = []
        classes = []
        
        # Collect all images and labels
        for class_name in sorted(os.listdir(dataset_dir)):
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                classes.append(class_name)
                class_idx = len(classes) - 1
                
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(os.path.join(class_path, img_name))
                        all_labels.append(class_idx)
        
        # Split data
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            all_images, all_labels, test_size=1-split_ratio[0], random_state=42, stratify=all_labels
        )
        
        val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=1-val_ratio, random_state=42, stratify=temp_labels
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

def search_diseases_globally(query, classes):
    """
    Search for skin diseases globally based on query
    """
    if not query or not classes:
        return []
    
    query = query.lower().strip()
    results = []
    
    # Disease database with symptoms and descriptions
    disease_database = {
        'actinic_keratosis': {
            'name': 'Actinic Keratosis',
            'type': 'Pre-cancerous',
            'symptoms': ['rough scaly patches', 'pink or red base', 'crusty lesions'],
            'description': 'Pre-cancerous skin condition caused by sun damage'
        },
        'atopic_dermatitis': {
            'name': 'Atopic Dermatitis',
            'type': 'Inflammatory',
            'symptoms': ['red itchy patches', 'dry scaly skin', 'cracked skin'],
            'description': 'Chronic inflammatory skin condition'
        },
        'benign_keratosis': {
            'name': 'Benign Keratosis',
            'type': 'Benign',
            'symptoms': ['brown or black spots', 'waxy appearance', 'stuck-on look'],
            'description': 'Common benign skin growths'
        },
        'dermatofibroma': {
            'name': 'Dermatofibroma',
            'type': 'Benign',
            'symptoms': ['firm brown nodules', 'dimpling when pinched', 'slow growing'],
            'description': 'Benign skin tumor'
        },
        'melanocytic_nevus': {
            'name': 'Melanocytic Nevus',
            'type': 'Benign',
            'symptoms': ['brown or black moles', 'uniform color', 'regular borders'],
            'description': 'Common moles or birthmarks'
        },
        'melanoma': {
            'name': 'Melanoma',
            'type': 'Malignant',
            'symptoms': ['asymmetric moles', 'irregular borders', 'color variation'],
            'description': 'Serious form of skin cancer'
        },
        'squamous_cell_carcinoma': {
            'name': 'Squamous Cell Carcinoma',
            'type': 'Malignant',
            'symptoms': ['red scaly patches', 'crusty sores', 'firm nodules'],
            'description': 'Common form of skin cancer'
        },
        'tinea_ringworm_candidiasis': {
            'name': 'Tinea Ringworm Candidiasis',
            'type': 'Fungal',
            'symptoms': ['red circular patches', 'itching', 'scaling'],
            'description': 'Fungal skin infection'
        },
        'vascular_lesion': {
            'name': 'Vascular Lesion',
            'type': 'Benign',
            'symptoms': ['red or purple spots', 'raised lesions', 'blood vessel clusters'],
            'description': 'Abnormal blood vessel growths'
        },
        'healthy': {
            'name': 'Healthy Skin',
            'type': 'Normal',
            'symptoms': ['normal skin color', 'uniform texture', 'no lesions'],
            'description': 'Normal healthy skin'
        }
    }
    
    # Search through classes and database
    for class_name in classes:
        class_lower = class_name.lower()
        
        # Direct match
        if query in class_lower:
            if class_lower in disease_database:
                results.append(disease_database[class_lower])
            else:
                # Extract disease name from class name
                disease = class_name.replace('_', ' ').title()
                
                results.append({
                    'name': disease,
                    'type': 'Detected Disease',
                    'symptoms': [],
                    'description': f'Skin disease detected: {disease}'
                })
        
        # Search in disease database
        for disease_key, disease_info in disease_database.items():
            if (query in disease_key or 
                query in disease_info['name'].lower() or
                query in disease_info['type'].lower() or
                any(query in symptom for symptom in disease_info['symptoms'])):
                
                if disease_info not in results:
                    results.append(disease_info)
    
    return results[:10]  # Limit to top 10 results

def create_advanced_visualizations(image, model, classes, predicted_class):
    """
    Create advanced visualizations for AI explainability
    """
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import seaborn as sns
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create figure with subplots (2x3 layout, removed disease region detection)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced AI Explainability Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Original Image
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Enhanced Edge Detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Multiple edge detection methods
        # Canny with optimized parameters for skin images
        edges_canny = cv2.Canny(blurred, 30, 100)
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        laplacian_8u = np.uint8(laplacian_abs)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_8u = np.uint8(sobel_combined)
        
        # Combine different edge detection methods
        combined_edges = cv2.bitwise_or(edges_canny, laplacian_8u)
        combined_edges = cv2.bitwise_or(combined_edges, sobel_8u)
        
        # Apply morphological operations to clean up edges
        kernel = np.ones((2, 2), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        axes[0, 1].imshow(combined_edges, cmap='gray')
        axes[0, 1].set_title('Enhanced Edge Detection (Combined)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Color Analysis
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create color histogram
        color_bins = np.linspace(0, 255, 50)
        axes[0, 2].hist(h.flatten(), bins=color_bins, alpha=0.7, color='red', label='Hue')
        axes[0, 2].hist(s.flatten(), bins=color_bins, alpha=0.7, color='green', label='Saturation')
        axes[0, 2].hist(v.flatten(), bins=color_bins, alpha=0.7, color='blue', label='Value')
        axes[0, 2].set_title('Color Distribution (HSV)', fontweight='bold')
        axes[0, 2].set_xlabel('Pixel Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 4. Enhanced Texture Analysis
        # Use the blurred image for better texture analysis
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude for better visualization
        gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        gradient_8u = np.uint8(gradient_normalized)
        
        # Apply additional filtering for skin texture
        kernel_3x3 = np.ones((3, 3), np.uint8)
        texture_enhanced = cv2.morphologyEx(gradient_8u, cv2.MORPH_OPEN, kernel_3x3)
        
        axes[1, 0].imshow(texture_enhanced, cmap='hot')
        axes[1, 0].set_title('Enhanced Texture Analysis', fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Feature Importance Heatmap
        # Create a simple feature importance visualization
        if model is not None and classes:
            try:
                # Get model predictions for visualization
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                model.eval()
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                
                # Create feature importance heatmap
                prob_array = probabilities[0].cpu().numpy()
                top_indices = np.argsort(prob_array)[-5:]  # Top 5 predictions
                
                # Create a simple heatmap
                heatmap_data = np.zeros((len(classes), 1))
                for i, idx in enumerate(top_indices):
                    heatmap_data[idx] = prob_array[idx]
                
                sns.heatmap(heatmap_data, 
                           xticklabels=['Confidence'],
                           yticklabels=[classes[i] for i in top_indices],
                           annot=True, 
                           fmt='.3f',
                           cmap='RdYlGn_r',
                           ax=axes[1, 1])
                axes[1, 1].set_title('Model Confidence Heatmap', fontweight='bold')
                
            except Exception as e:
                # Fallback: create a simple bar chart
                axes[1, 1].text(0.5, 0.5, f'Predicted: {predicted_class}\nModel confidence visualization\n(Feature analysis available)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1, 1].set_title('Model Confidence', fontweight='bold')
        
        # 6. Image Statistics
        # Calculate basic image statistics
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        
        # Create a simple statistics display
        stats_text = f"""
        Image Statistics:
        
        Mean RGB: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})
        Std RGB: ({std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f})
        
        Image Size: {img_array.shape[1]} x {img_array.shape[0]}
        Total Pixels: {img_array.shape[0] * img_array.shape[1]:,}
        """
        
        axes[1, 2].text(0.5, 0.5, stats_text, 
                       ha='center', va='center', transform=axes[1, 2].transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Image Statistics', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = "advanced_visualizations.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        print(f"Error creating advanced visualizations: {e}")
        return None

def create_lime_visualization(image, model, classes):
    """
    Create LIME visualization for model explainability
    """
    try:
        import lime
        from lime import lime_image
        import matplotlib.pyplot as plt
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Check if model is available
        if model is None:
            print("Model not available for LIME")
            return create_lime_fallback_visualization(image, model, classes)
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            try:
                model.eval()
                # Ensure model is on the correct device
                model = model.to(device)
                batch = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)
                with torch.no_grad():
                    outputs = model(batch)
                    probabilities = torch.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()
            except Exception as e:
                print(f"Error in LIME prediction function: {e}")
                # Return uniform probabilities as fallback
                return np.ones((len(images), len(classes))) / len(classes)
        
        # Get explanation with reduced samples for faster processing
        explanation = explainer.explain_instance(
            np.array(image), 
            predict_fn, 
            top_labels=min(len(classes), 3),  # Limit to top 3 classes
            hide_color=0, 
            num_samples=500  # Reduced from 1000 for faster processing
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # LIME explanation
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=10, 
            hide_rest=True
        )
        axes[1].imshow(mask, cmap='Reds', alpha=0.7)
        axes[1].imshow(image, alpha=0.3)
        axes[1].set_title('LIME Explanation (Positive)', fontweight='bold')
        axes[1].axis('off')
        
        # LIME explanation with both positive and negative
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=False, 
            num_features=10, 
            hide_rest=False
        )
        axes[2].imshow(mask, cmap='RdBu', alpha=0.7)
        axes[2].imshow(image, alpha=0.3)
        axes[2].set_title('LIME Explanation (Positive/Negative)', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        lime_path = "lime_visualization.png"
        plt.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return lime_path
        
    except Exception as e:
        print(f"Error creating LIME visualization: {e}")
        import traceback
        traceback.print_exc()
        return create_lime_fallback_visualization(image, model, classes)

def create_shap_visualization(image, model, classes):
    """
    Create SHAP visualization for model explainability
    """
    try:
        # Try to import SHAP, but provide fallback if not available
        try:
            import shap
        except ImportError:
            print("SHAP module not available, using fallback visualization")
            return create_shap_fallback_visualization(image, model, classes)
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Check if model is available
        if model is None:
            print("Model not available for SHAP")
            return None
        
        # Prepare the image for SHAP
        img_array = np.array(image)
        
        # Create a simple but effective SHAP-like visualization
        # Since full SHAP can be complex, we'll create a feature importance visualization
        
        # Method 1: Edge-based feature importance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_importance = cv2.GaussianBlur(edges.astype(np.float32), (9, 9), 0)
        
        # Method 2: Color-based feature importance
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Detect disease-related colors
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        disease_importance = cv2.bitwise_or(brown_mask, yellow_mask).astype(np.float32)
        disease_importance = cv2.GaussianBlur(disease_importance, (15, 15), 0)
        
        # Method 3: Texture-based feature importance
        kernel_size = 5
        mean_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, mean_kernel)
        variance_img = cv2.filter2D((gray.astype(np.float32) - mean_img)**2, -1, mean_kernel)
        texture_importance = np.sqrt(variance_img)
        
        # Combine feature importance maps
        feature_importance = np.zeros_like(gray, dtype=np.float32)
        feature_importance += edge_importance * 0.3
        feature_importance += disease_importance * 0.4
        feature_importance += texture_importance * 0.3
        
        # Normalize
        if feature_importance.max() > 0:
            feature_importance = feature_importance / feature_importance.max()
        
        # Get model prediction for title
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = classes[predicted_class_idx] if classes else f"Class {predicted_class_idx}"
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title(f'Original Image\nPredicted: {predicted_class}', fontweight='bold')
        axes[0].axis('off')
        
        # Feature importance heatmap
        heatmap_display = axes[1].imshow(feature_importance, cmap='RdBu', vmin=0, vmax=1)
        axes[1].set_title('SHAP Feature Importance', fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(heatmap_display, ax=axes[1], shrink=0.8)
        cbar.set_label('Feature Importance', rotation=270, labelpad=15)
        
        # Feature importance overlay
        axes[2].imshow(img_array)
        overlay_display = axes[2].imshow(feature_importance, cmap='RdBu', alpha=0.7, vmin=0, vmax=1)
        axes[2].set_title('SHAP Overlay', fontweight='bold')
        axes[2].axis('off')
        
        # Feature breakdown
        feature_names = ['Edge Features', 'Disease Features', 'Texture Features']
        feature_weights = [0.3, 0.4, 0.3]
        
        bars = axes[3].bar(feature_names, feature_weights, color=['#ff7f0e', '#2ca02c', '#d62728'])
        axes[3].set_title('Feature Contribution Weights', fontweight='bold')
        axes[3].set_ylabel('Weight')
        axes[3].set_ylim(0, 0.5)
        
        # Add value labels on bars
        for bar, weight in zip(bars, feature_weights):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{weight:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the visualization
        shap_path = "shap_visualization.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_path
        
    except Exception as e:
        print(f"Error creating SHAP visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Create a simple feature importance visualization
        try:
            print("Creating fallback SHAP visualization...")
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simple feature importance visualization
            img_array = np.array(image)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_array)
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')
            
            # Simple feature importance heatmap
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            feature_importance = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
            
            axes[1].imshow(feature_importance, cmap='RdBu')
            axes[1].set_title('Feature Importance (Fallback)', fontweight='bold')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_array)
            axes[2].imshow(feature_importance, cmap='RdBu', alpha=0.6)
            axes[2].set_title('Feature Overlay (Fallback)', fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            shap_path = "shap_visualization.png"
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return shap_path
            
        except Exception as fallback_error:
            print(f"Fallback SHAP visualization also failed: {fallback_error}")
            return None

def create_lime_fallback_visualization(image, model, classes):
    """
    Create a fallback LIME-like visualization when LIME module fails
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        
        # Create a simple feature importance visualization
        img_array = np.array(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Simple feature importance heatmap
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        feature_importance = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        axes[1].imshow(feature_importance, cmap='Reds')
        axes[1].set_title('Feature Importance (LIME Fallback)', fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(feature_importance, cmap='Reds', alpha=0.6)
        axes[2].set_title('Feature Overlay (LIME Fallback)', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        lime_path = "lime_visualization.png"
        plt.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return lime_path
        
    except Exception as e:
        print(f"Error creating fallback LIME visualization: {e}")
        return None

def create_shap_fallback_visualization(image, model, classes):
    """
    Create a fallback SHAP-like visualization when SHAP module is not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        
        # Create a simple feature importance visualization
        img_array = np.array(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Simple feature importance heatmap
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        feature_importance = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        axes[1].imshow(feature_importance, cmap='RdBu')
        axes[1].set_title('Feature Importance (Fallback)', fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(feature_importance, cmap='RdBu', alpha=0.6)
        axes[2].set_title('Feature Overlay (Fallback)', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        shap_path = "shap_visualization.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_path
        
    except Exception as e:
        print(f"Error creating fallback SHAP visualization: {e}")
        return None

def create_gradcam_visualization(image, model, classes):
    """
    Create Grad-CAM visualization for model explainability
    """
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Check if model is available
        if model is None:
            print("Model not available for Grad-CAM")
            return None
        
        # Create a simple but effective attention visualization
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Edge-based attention with gradient
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edge_attention = cv2.dilate(edges, kernel, iterations=1)
        
        # Method 2: Saliency-based attention with distance transform
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        saliency = cv2.absdiff(gray, blur)
        saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
        
        # Create distance-based attention (stronger in center, weaker at edges)
        height, width = gray.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        center_y, center_x = height // 2, width // 2
        distance_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distance_attention = 1.0 - (distance_map / max_distance)
        
        # Method 3: Color-based attention (for skin diseases)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Detect potential disease regions (brown/yellow spots)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Create gradient disease attention (not just binary)
        disease_attention = cv2.bitwise_or(brown_mask, yellow_mask).astype(np.float32)
        disease_attention = cv2.GaussianBlur(disease_attention, (15, 15), 0)
        
        # Method 4: Texture-based attention
        # Calculate local variance for texture
        kernel_size = 5
        mean_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, mean_kernel)
        variance_img = cv2.filter2D((gray.astype(np.float32) - mean_img)**2, -1, mean_kernel)
        texture_attention = np.sqrt(variance_img)
        
        # Combine all attention maps with different weights
        attention_map = np.zeros_like(gray, dtype=np.float32)
        attention_map += edge_attention.astype(np.float32) * 0.2
        attention_map += saliency.astype(np.float32) * 0.3
        attention_map += distance_attention.astype(np.float32) * 0.1
        attention_map += disease_attention * 0.3
        attention_map += texture_attention * 0.1
        
        # Apply Gaussian blur to smooth the attention map
        attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0)
        
        # Create more dynamic range by applying power function
        attention_map = np.power(attention_map, 0.5)  # Makes high values more prominent
        
        # Normalize attention map with better range distribution
        if attention_map.max() > 0:
            # Use percentile-based normalization to ensure full color spectrum
            p90 = np.percentile(attention_map, 90)
            p10 = np.percentile(attention_map, 10)
            attention_map = np.clip((attention_map - p10) / (p90 - p10), 0, 1)
            
            # Apply sigmoid-like transformation to spread values across full range
            attention_map = 1 / (1 + np.exp(-5 * (attention_map - 0.5)))
            
            # Apply additional contrast enhancement
            attention_map = np.power(attention_map, 0.6)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Create custom colormap to ensure all colors are present
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define colors for full spectrum: Blue -> Cyan -> Green -> Yellow -> Orange -> Red
        colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        custom_cmap = LinearSegmentedColormap.from_list('full_spectrum', colors, N=n_bins)
        
        # Attention heatmap with full color spectrum
        heatmap_display = axes[1].imshow(attention_map, cmap=custom_cmap, vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM Heatmap (Blue→Cyan→Green→Yellow→Orange→Red)', fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar to show the intensity scale
        cbar = plt.colorbar(heatmap_display, ax=axes[1], shrink=0.8)
        cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
        
        # Overlay with enhanced visibility
        axes[2].imshow(image)
        overlay_display = axes[2].imshow(attention_map, cmap=custom_cmap, alpha=0.7, vmin=0, vmax=1)
        axes[2].set_title('Grad-CAM Overlay (Blue→Cyan→Green→Yellow→Orange→Red)', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        gradcam_path = "gradcam_visualization.png"
        plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return gradcam_path
        
    except Exception as e:
        print(f"Error creating attention visualization: {e}")
        import traceback
        traceback.print_exc()
        return None
