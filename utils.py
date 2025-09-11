"""
Blood Cell Detection Utilities - BloodCellAI
Utility functions for blood smear analysis, cell detection, and report generation
"""

# Standard library imports
import os
import time
import tempfile
import glob
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Handle all optional imports with try-except
REQUIRED_PACKAGES = {}

try:
    import numpy as np
    REQUIRED_PACKAGES['numpy'] = True
except ImportError:
    REQUIRED_PACKAGES['numpy'] = False
    print("Warning: numpy not found. Install with: pip install numpy")

try:
    import cv2
    REQUIRED_PACKAGES['cv2'] = True
except ImportError:
    REQUIRED_PACKAGES['cv2'] = False
    print("Warning: opencv-python not found. Install with: pip install opencv-python-headless")

try:
    import torch
    from torchvision import transforms
    REQUIRED_PACKAGES['torch'] = True
except ImportError:
    REQUIRED_PACKAGES['torch'] = False
    print("Warning: torch/torchvision not found. Install with: pip install torch torchvision")

try:
    from PIL import Image, ImageFilter, ImageEnhance
    REQUIRED_PACKAGES['PIL'] = True
except ImportError:
    REQUIRED_PACKAGES['PIL'] = False
    print("Warning: Pillow not found. Install with: pip install Pillow")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    REQUIRED_PACKAGES['viz'] = True
except ImportError:
    REQUIRED_PACKAGES['viz'] = False
    print("Warning: matplotlib/seaborn not found. Install with: pip install matplotlib seaborn")
    
def generate_advanced_report(detections: Dict, image_path: str) -> Dict:
    """
    Generate an advanced report with detailed statistics and visualizations
    
    Args:
        detections: Detection results from enhance_cell_detection
        image_path: Path to the original image
        
    Returns:
        Dictionary containing report data and file paths
    """
    try:
        # Create report directory if it doesn't exist
        report_dir = "evaluation_results"
        os.makedirs(report_dir, exist_ok=True)
        
        # Extract statistics
        stats = detections['stats']
        
        # Generate detailed text report
        report_text = f"""Blood Cell Detection Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Summary:
        - Total Cells Detected: {stats['total_cells_detected']}
        - RBC Count: {stats['RBC_count']}
        - WBC Count: {stats['WBC_count']}
        - Platelet Count: {stats['Platelet_count']}
        
        Cell Distribution:
        - RBC: {stats['cell_distribution']['RBC_percentage']:.2f}%
        - WBC: {stats['cell_distribution']['WBC_percentage']:.2f}%
        - Platelets: {stats['cell_distribution']['Platelet_percentage']:.2f}%
        
        Confidence Scores:
        - RBC: {stats['confidence_scores']['RBC']:.3f}
        - WBC: {stats['confidence_scores']['WBC']:.3f}
        - Platelets: {stats['confidence_scores']['Platelets']:.3f}
        - Overall: {stats['confidence_scores']['Overall']:.3f}
        """
        
        # Save detailed report
        report_path = os.path.join(report_dir, "detailed_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Create visualizations
        plt.style.use('seaborn')
        
        # Cell distribution pie chart
        plt.figure(figsize=(10, 8))
        cell_types = ['RBC', 'WBC', 'Platelets']
        cell_counts = [stats['RBC_count'], stats['WBC_count'], stats['Platelet_count']]
        plt.pie(cell_counts, labels=cell_types, autopct='%1.1f%%', colors=sns.color_palette("husl", 3))
        plt.title('Cell Type Distribution')
        plt.savefig(os.path.join(report_dir, 'cell_distribution_pie.png'))
        plt.close()
        
        # Cell counts bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(cell_types, cell_counts)
        plt.title('Cell Counts by Type')
        plt.ylabel('Count')
        plt.savefig(os.path.join(report_dir, 'cell_metrics_bar.png'))
        plt.close()
        
        # Confidence scores radar chart
        conf_scores = [stats['confidence_scores'][cell_type] for cell_type in cell_types]
        angles = np.linspace(0, 2*np.pi, len(cell_types), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        conf_scores = np.concatenate((conf_scores, [conf_scores[0]]))
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, conf_scores)
        ax.fill(angles, conf_scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cell_types)
        plt.title('Detection Confidence by Cell Type')
        plt.savefig(os.path.join(report_dir, 'detection_confidence_radar.png'))
        plt.close()
        
        return {
            'report_path': report_path,
            'visualization_paths': {
                'distribution_pie': os.path.join(report_dir, 'cell_distribution_pie.png'),
                'metrics_bar': os.path.join(report_dir, 'cell_metrics_bar.png'),
                'confidence_radar': os.path.join(report_dir, 'detection_confidence_radar.png')
            }
        }
        
    except Exception as e:
        print(f"Error generating advanced report: {str(e)}")
        return None

def save_cell_specific_images(visualizations: Dict[str, np.ndarray], base_filename: str) -> Dict[str, str]:
    """
    Save cell-specific visualization images
    
    Args:
        visualizations: Dictionary containing cell-specific visualization arrays
        base_filename: Base name for the output files
        
    Returns:
        Dictionary containing paths to saved images
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join("evaluation_results", "cell_specific_images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each visualization
        saved_paths = {}
        for cell_type, image in visualizations.items():
            output_path = os.path.join(output_dir, f"{base_filename}_{cell_type}.png")
            cv2.imwrite(output_path, image)
            saved_paths[cell_type] = output_path
            
        return saved_paths
        
    except Exception as e:
        print(f"Error saving cell-specific images: {str(e)}")
        return None
    except ImportError:
      REQUIRED_PACKAGES['viz'] = False
    print("Warning: matplotlib/seaborn not found. Install with: pip install matplotlib seaborn")

try:
    import streamlit as st
    REQUIRED_PACKAGES['streamlit'] = True
except ImportError:
    REQUIRED_PACKAGES['streamlit'] = False
    print("Warning: streamlit not found. Install with: pip install streamlit")

# Explainability packages removed - focusing on core detection

# Additional ML imports
try:
    import pandas as pd
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    REQUIRED_PACKAGES['ml'] = True
except ImportError:
    REQUIRED_PACKAGES['ml'] = False
    print("Warning: pandas/scikit-learn not found. Install with: pip install pandas scikit-learn")

# Report generation
try:
    from fpdf import FPDF
except ImportError:
    print("Warning: fpdf not found. Install with: pip install fpdf2")

# Constants for blood cell detection
BLOOD_CELL_CLASSES = ['Platelets', 'RBC', 'WBC']
REQUIRED_DATASET_STRUCTURE = ['train', 'valid', 'test']
REQUIRED_SUBFOLDERS = ['images', 'labels']

def validate_dataset(dataset_path: str) -> tuple[bool, str]:
    """
    Validates the blood cell detection dataset structure.
    
    Args:
        dataset_path (str): Path to the dataset root directory
        
    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    if not os.path.exists(dataset_path):
        return False, f"Dataset directory '{dataset_path}' does not exist"
        
    # Check for required splits
    for split in REQUIRED_DATASET_STRUCTURE:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            return False, f"Missing {split} directory"
            
        # Check for images and labels folders
        for subfolder in REQUIRED_SUBFOLDERS:
            subfolder_path = os.path.join(split_path, subfolder)
            if not os.path.exists(subfolder_path):
                return False, f"Missing {subfolder} folder in {split} directory"
                
            # Check if folders are not empty
            if len(os.listdir(subfolder_path)) == 0:
                return False, f"No files found in {split}/{subfolder}"
                
    # Validate data.yaml
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        return False, "Missing data.yaml configuration file"
        
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'names' not in config:
                return False, "Missing 'names' field in data.yaml"
            if not isinstance(config['names'], (list, dict)):
                return False, "Invalid format for 'names' in data.yaml"
            if isinstance(config['names'], dict):
                classes = list(config['names'].values())
            else:
                classes = config['names']
            if not all(c in BLOOD_CELL_CLASSES for c in classes):
                return False, f"Invalid cell types. Expected: {BLOOD_CELL_CLASSES}"
    except Exception as e:
        return False, f"Error reading data.yaml: {str(e)}"
        
    return True, "Dataset validation successful"

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

def check_requirements():
    """Check if all required packages are installed"""
    missing_packages = [pkg for pkg, available in REQUIRED_PACKAGES.items() if not available]
    if missing_packages:
        print("Warning: The following required packages are missing:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages with:")
        print("pip install numpy opencv-python-headless torch torchvision Pillow streamlit matplotlib seaborn scikit-learn pandas")
        return False
    return True

def check_image_quality(image: Image.Image) -> float:
    """
    Check image quality for blood cell detection
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

def describe_image(image: Image.Image, cell_type: str = None) -> str:
    """
    Generate detailed description of blood smear image for cell analysis
    """
    try:
        processor, model = load_models()
        if processor is None or model is None:
            return "Blood smear image showing cellular structures"
        
        # Prepare image for BLIP
        inputs = processor(image, return_tensors="pt")
        
        # Generate description
        out = model.generate(**inputs, max_length=100, num_beams=5)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance description for blood cell analysis
        enhanced_description = f"Blood smear image showing: {description}. "
        
        # Add blood-specific details
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Analyze color distribution for blood cells
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze cellular structures
        mean_color = np.mean(img_array, axis=(0, 1))
        color_variance = np.var(img_array, axis=(0, 1))
        
        # Detect cellular structures
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for cellular structures
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        if edge_density > 0.05:
            enhanced_description += "Image shows distinct cellular structures and boundaries. "
        else:
            enhanced_description += "Image shows uniform cellular background. "
        
        # Analyze color variations for different cell types
        red_channel = np.mean(img_array[:,:,0])
        blue_channel = np.mean(img_array[:,:,2])
        
        if red_channel > blue_channel * 1.2:
            enhanced_description += "Image shows red-dominant coloration typical of RBC-rich areas. "
        elif blue_channel > red_channel * 1.1:
            enhanced_description += "Image shows blue-purple coloration typical of nucleated cells. "
        else:
            enhanced_description += "Image shows balanced coloration with mixed cell populations. "
        
        # Check for circular structures (typical of blood cells)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)
        if circles is not None:
            enhanced_description += f"Image contains approximately {len(circles[0])} circular structures consistent with blood cells. "
        
        return enhanced_description
        
    except Exception as e:
        print(f"Error describing image: {e}")
        return "Blood smear image for cellular analysis"

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
                from langchain_groq import ChatGroq
                
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test blood cell analysis")
                    if test_response:
                        return True, f"Working (using {model_name})"
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                result = test_model()
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

def load_css():
    """Load custom CSS for blood cell detection app"""
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
    
    .compact-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
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
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)

def get_image_transform():
    """Get image transformation for blood cell detection"""
    return transforms.Compose([
        transforms.Resize((640, 640)),  # YOLO typical input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])

def gradient_text(text, color1, color2):
    """Create gradient text effect"""
    return f'<span style="background: linear-gradient(45deg, {color1}, {color2}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;">{text}</span>'

def plot_cell_distribution(cell_counts):
    """Plot cell distribution chart"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cell_types = list(cell_counts.keys())
        counts = list(cell_counts.values())
        colors = ['#ef5350', '#42a5f5', '#66bb6a']
        
        bars = ax.bar(cell_types, counts, color=colors)
        ax.set_title('Blood Cell Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting cell distribution: {e}")
        return None

def generate_report(detection_results: Dict[str, Any], agent: Any = None) -> str:
    """
    Generate a comprehensive analysis report from detection results.
    
    Args:
        detection_results: Dictionary containing detection statistics and other data.
        agent: The AI agent instance for generating the report.
        
    Returns:
        str: A formatted string containing the full analysis report.
    """
    try:
        stats = detection_results.get('stats', {})
        
        if agent:
            try:
                # Prepare data for the agent
                image_description = "A standard blood smear image." # Placeholder
                morphology_notes = "Morphology appears generally normal, detailed analysis pending." # Placeholder
                
                # Call the agent to get the detailed report
                analysis_result = agent.analyze_blood_sample(
                    image_description=image_description,
                    detected_cells=['RBC', 'WBC', 'Platelets'],
                    confidences=[
                        stats.get('confidence_scores', {}).get('RBC', 0),
                        stats.get('confidence_scores', {}).get('WBC', 0),
                        stats.get('confidence_scores', {}).get('Platelets', 0)
                    ],
                    count_data=stats,
                    morphology=morphology_notes
                )
                
                if "analysis" in analysis_result:
                    return analysis_result["analysis"]
                else:
                    # Fallback to basic report if agent fails
                    print("Agent failed to generate report, creating basic version.")
                    return generate_basic_report(stats)

            except Exception as agent_error:
                print(f"Error calling AI agent: {agent_error}")
                return generate_basic_report(stats)
        else:
            # Generate basic report if no agent is provided
            return generate_basic_report(stats)

    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error: Could not generate the analysis report."

def generate_basic_report(stats: Dict[str, Any]) -> str:
    """Generates a basic, data-driven report when the AI agent is unavailable."""
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Basic clinical status
    rbc_count = stats.get('RBC_count', 0)
    wbc_count = stats.get('WBC_count', 0)
    platelet_count = stats.get('Platelet_count', 0)
    
    rbc_status = "Normal"
    if rbc_count < 4.5: rbc_status = "Low"
    elif rbc_count > 5.5: rbc_status = "High"
    
    wbc_status = "Normal"
    if wbc_count < 4500: wbc_status = "Low"
    elif wbc_count > 11000: wbc_status = "High"
    
    platelet_status = "Normal"
    if platelet_count < 150000: platelet_status = "Low"
    elif platelet_count > 450000: platelet_status = "High"

    report = f"""
    # Basic Blood Cell Analysis Report
    **Generated on:** {report_date}
    
    ## 1. Detection Summary
    - **Total Cells Detected:** {stats.get('total_cells_detected', 0)}
    - **Overall Confidence:** {stats.get('confidence_scores', {}).get('Overall', 0):.2%}
    
    ## 2. Individual Cell Counts
    - **Red Blood Cells (RBC):** {rbc_count:,}
    - **White Blood Cells (WBC):** {wbc_count:,}
    - **Platelets:** {platelet_count:,}
    
    ## 3. Cell Distribution
    - **RBC Percentage:** {stats.get('cell_distribution', {}).get('RBC_percentage', 0):.1f}%
    - **WBC Percentage:** {stats.get('cell_distribution', {}).get('WBC_percentage', 0):.1f}%
    - **Platelet Percentage:** {stats.get('cell_distribution', {}).get('Platelet_percentage', 0):.1f}%
    
    ## 4. Clinical Interpretation
    - **RBC Status:** {rbc_status}
    - **WBC Status:** {wbc_status}
    - **Platelet Status:** {platelet_status}
    
    **Note:** This is a data-only report. The AI agent was unavailable for detailed narrative analysis.
    """
    return report

class BloodCellPDF(FPDF):
    """PDF generator for blood cell analysis reports"""
    
    def __init__(self, blood_info=""):
        super().__init__()
        self.blood_info = self.sanitize_text(blood_info)
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
        self.cell(0, 10, 'Blood Cell Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Footer for each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def cover_page(self):
        """Create cover page"""
        self.set_font('Arial', 'B', 24)
        self.cell(0, 60, 'Blood Cell Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 20, 'AI-Powered Hematological Assessment', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 20, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        if self.blood_info:
            sanitized_info = self.sanitize_text(self.blood_info)
            self.cell(0, 20, f'Sample Information: {sanitized_info}', 0, 1, 'C')
        self.add_page()
    
    def table_of_contents(self):
        """Add table of contents"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(5)
        
        sections = [
            'Executive Summary',
            'Image Analysis',
            'Cell Detection Results',
            'Cell Count Analysis',
            'Morphological Assessment',
            'Clinical Interpretation',
            'Quality Assessment',
            'Follow-up Recommendations'
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
    
    def add_cell_count_table(self, cell_counts):
        """Add cell count table"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Cell Count Summary', 0, 1, 'L')
        self.ln(5)
        
        # Table headers
        self.set_font('Arial', 'B', 12)
        self.cell(60, 8, 'Cell Type', 1, 0, 'C')
        self.cell(40, 8, 'Count', 1, 0, 'C')
        self.cell(80, 8, 'Normal Range', 1, 1, 'C')
        
        # Table data
        self.set_font('Arial', '', 11)
        normal_ranges = {
            'RBC': '4.5-5.5 million/uL',
            'WBC': '4,500-11,000/uL',
            'Platelets': '150,000-450,000/uL'
        }
        
        for cell_type, count in cell_counts.items():
            self.cell(60, 8, cell_type, 1, 0, 'L')
            self.cell(40, 8, str(count), 1, 0, 'C')
            self.cell(80, 8, normal_ranges.get(cell_type, 'N/A'), 1, 1, 'C')
        
        self.ln(10)
    
    def add_summary(self, report, blood_context=None):
        """Add executive summary"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        
        # Extract key information from report
        summary_points = [
            "AI-powered blood cell detection completed successfully",
            "Comprehensive analysis of cellular morphology and counts",
            "Detailed hematological assessment provided",
            "Quality assurance and follow-up recommendations outlined"
        ]
        
        for point in summary_points:
            self.cell(0, 5, f"- {point}", 0, 1, 'L')
        
        if blood_context:
            self.ln(5)
            sanitized_context = self.sanitize_text(blood_context)
            self.cell(0, 5, f"Sample Context: {sanitized_context}", 0, 1, 'L')
        
        self.ln(10)

class BloodCellPDF(FPDF):
    """PDF generator for blood cell analysis reports"""
    
    def __init__(self, blood_info=""):
        super().__init__()
        self.blood_info = self.sanitize_text(blood_info)
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
        self.cell(0, 10, 'Blood Cell Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Footer for each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def cover_page(self):
        """Create cover page"""
        self.set_font('Arial', 'B', 24)
        self.cell(0, 60, 'Blood Cell Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 20, 'AI-Powered Hematological Assessment', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 20, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        if self.blood_info:
            sanitized_info = self.sanitize_text(self.blood_info)
            self.cell(0, 20, f'Sample Information: {sanitized_info}', 0, 1, 'C')
        self.add_page()
    
    def table_of_contents(self):
        """Add table of contents"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(5)
        
        sections = [
            'Executive Summary',
            'Image Analysis',
            'Cell Detection Results',
            'Cell Count Analysis',
            'Morphological Assessment',
            'Clinical Interpretation',
            'Quality Assessment',
            'Follow-up Recommendations'
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
    
    def add_cell_count_table(self, cell_counts):
        """Add cell count table"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Cell Count Summary', 0, 1, 'L')
        self.ln(5)
        
        # Table headers
        self.set_font('Arial', 'B', 12)
        self.cell(60, 8, 'Cell Type', 1, 0, 'C')
        self.cell(40, 8, 'Count', 1, 0, 'C')
        self.cell(80, 8, 'Normal Range', 1, 1, 'C')
        
        # Table data
        self.set_font('Arial', '', 11)
        normal_ranges = {
            'RBC': '4.5-5.5 million/uL',
            'WBC': '4,500-11,000/uL',
            'Platelets': '150,000-450,000/uL'
        }
        
        for cell_type, count in cell_counts.items():
            self.cell(60, 8, cell_type, 1, 0, 'L')
            self.cell(40, 8, str(count), 1, 0, 'C')
            self.cell(80, 8, normal_ranges.get(cell_type, 'N/A'), 1, 1, 'C')
        
        self.ln(10)
    
    def add_summary(self, report, blood_context=None):
        """Add executive summary"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Arial', '', 11)
        
        # Extract key information from report
        summary_points = [
            "AI-powered blood cell detection completed successfully",
            "Comprehensive analysis of cellular morphology and counts",
            "Detailed hematological assessment provided",
            "Quality assurance and follow-up recommendations outlined"
        ]
        
        for point in summary_points:
            self.cell(0, 5, f"- {point}", 0, 1, 'L')
        
        if blood_context:
            self.ln(5)
            sanitized_context = self.sanitize_text(blood_context)
            self.cell(0, 5, f"Sample Context: {sanitized_context}", 0, 1, 'L')
        
        self.ln(10)