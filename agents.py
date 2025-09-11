"""
AI Agents for BloodCellAI - Blood Cell Analysis Platform
Enhanced with intelligent agents for comprehensive blood cell analysis
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import time
import random

# LangChain imports
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure model settings
MODEL_NAME = "gemma2-9b-it"
MODEL_TEMPERATURE = 0.1

def retry_with_exponential_backoff(func, max_retries=5, base_delay=2, max_delay=30):
    """
    Enhanced retry mechanism with exponential backoff and better error handling.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries (default: 5)
        base_delay: Base delay in seconds (default: 2)
        max_delay: Maximum delay between retries in seconds (default: 30)
    
    Returns:
        Result of the function call
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Define retryable errors
            retryable_conditions = [
                "over capacity",
                "503",
                "timeout",
                "connection error",
                "network",
                "too many requests",
                "temporary failure"
            ]
            
            # Check if error is retryable
            should_retry = any(condition in error_msg for condition in retryable_conditions)
            
            if not should_retry:
                print(f"Non-retryable error: {error_msg}")
                raise e
            
            if attempt == max_retries:
                print(f"Max retries ({max_retries}) reached. Last error: {error_msg}")
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
            print(f"Retry attempt {attempt + 1}/{max_retries + 1}. Waiting {delay:.2f} seconds...")
            time.sleep(delay)

# Blood cell knowledge base
BLOOD_CELL_KNOWLEDGE = {
    "RBC": {
        "characteristics": ["round biconcave disk", "red color", "no nucleus"],
        "normal_count": "4.5-5.5 million cells/microliter",
        "function": ["oxygen transport", "carbon dioxide transport"],
        "abnormalities": {
            "low": ["anemia", "blood loss", "malnutrition"],
            "high": ["polycythemia", "dehydration", "lung disease"]
        },
        "severity_indicators": {
            "shape": ["normal", "sickle", "spherical", "elliptical"],
            "size": ["normal", "microcytic", "macrocytic"],
            "color": ["normal", "hypochromic", "hyperchromic"]
        }
    },
    "WBC": {
        "characteristics": ["has nucleus", "irregular shape", "larger than RBC"],
        "normal_count": "4,500-11,000 cells/microliter",
        "types": {
            "neutrophils": {
                "appearance": "multi-lobed nucleus",
                "function": "bacterial infection defense"
            },
            "lymphocytes": {
                "appearance": "round nucleus",
                "function": "immune response"
            },
            "monocytes": {
                "appearance": "kidney-shaped nucleus",
                "function": "phagocytosis"
            },
            "eosinophils": {
                "appearance": "bi-lobed nucleus",
                "function": "allergy response"
            },
            "basophils": {
                "appearance": "s-shaped nucleus",
                "function": "inflammatory response"
            }
        },
        "abnormalities": {
            "low": ["immunodeficiency", "bone marrow problems"],
            "high": ["infection", "inflammation", "leukemia"]
        }
    },
    "Platelets": {
        "characteristics": ["small fragments", "no nucleus", "irregular shape"],
        "normal_count": "150,000-450,000 per microliter",
        "function": ["blood clotting", "wound healing"],
        "abnormalities": {
            "low": ["bleeding disorders", "bone marrow failure"],
            "high": ["thrombocytosis", "inflammation"]
        },
        "severity_indicators": {
            "count": ["normal", "low", "high"],
            "size": ["normal", "large", "small"],
            "distribution": ["normal", "clumped", "scattered"]
        }
    }
}

class BloodCellAnalysisTool(BaseTool):
    name: str = "blood_cell_analyzer"
    description: str = "Analyze blood cell types and their characteristics"
    return_direct: bool = True
    
    def _run(self, image_description: str, detected_cell: str, confidence: float) -> str:
        """Analyze blood cell image and provide detailed insights"""
        analysis = {
            "cell_type": detected_cell,
            "confidence": confidence,
            "characteristics": self._get_characteristics(detected_cell),
            "abnormalities": self._check_abnormalities(detected_cell, image_description),
            "recommendations": self._get_recommendations(detected_cell),
            "count_assessment": self._assess_count(detected_cell, confidence)
        }
        return json.dumps(analysis, indent=2)
    
    def _get_characteristics(self, cell_type: str) -> Dict:
        cell_lower = cell_type.lower()
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            return {
                "features": BLOOD_CELL_KNOWLEDGE[cell_lower]["characteristics"],
                "normal_count": BLOOD_CELL_KNOWLEDGE[cell_lower]["normal_count"],
                "function": BLOOD_CELL_KNOWLEDGE[cell_lower].get("function", [])
            }
        return {"error": "Cell type not found in knowledge base"}
    
    def _check_abnormalities(self, cell_type: str, description: str) -> Dict:
        cell_lower = cell_type.lower()
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            abnormalities = BLOOD_CELL_KNOWLEDGE[cell_lower].get("abnormalities", {})
            severity = self._assess_severity(cell_type, description)
            return {
                "possible_conditions": abnormalities,
                "severity": severity
            }
        return {"error": "Unable to assess abnormalities"}
    
    def _get_recommendations(self, cell_type: str) -> List[str]:
        cell_lower = cell_type.lower()
        recommendations = []
        
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            cell_info = BLOOD_CELL_KNOWLEDGE[cell_lower]
            recommendations.extend([
                f"Monitor {cell_type} count and morphology",
                f"Normal range: {cell_info['normal_count']}",
                "Consider complete blood count (CBC) test"
            ])
        
        recommendations.append("Consult hematologist for detailed analysis")
        return recommendations
    
    def _assess_severity(self, cell_type: str, description: str) -> str:
        cell_lower = cell_type.lower()
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            indicators = BLOOD_CELL_KNOWLEDGE[cell_lower].get("severity_indicators", {})
            description_lower = description.lower()
            
            # Check for known abnormal indicators in the description
            abnormal_indicators = sum(
                any(indicator in description_lower for indicator in values[1:])  # Skip 'normal'
                for values in indicators.values()
            )
            
            if abnormal_indicators > 2:
                return "High"
            elif abnormal_indicators > 0:
                return "Moderate"
        
        return "Normal"
    
    def _assess_count(self, cell_type: str, confidence: float) -> Dict:
        cell_lower = cell_type.lower()
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            return {
                "normal_range": BLOOD_CELL_KNOWLEDGE[cell_lower]["normal_count"],
                "detection_confidence": confidence,
                "recommendation": "Verify with manual count if needed"
            }
        return {"error": "Unable to assess cell count"}

class CellMorphologyTool(BaseTool):
    name: str = "morphology_analyzer"
    description: str = "Analyze blood cell morphology and characteristics"
    
    def _run(self, cell_type: str, morphology_description: str) -> str:
        """Analyze blood cell morphology"""
        cell_lower = cell_type.lower()
        
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            cell_info = BLOOD_CELL_KNOWLEDGE[cell_lower]
            analysis = self._analyze_morphology(cell_type, morphology_description, cell_info)
            return json.dumps(analysis, indent=2)
        
        return "Cell type not found in knowledge base"
    
    def _analyze_morphology(self, cell_type: str, description: str, cell_info: Dict) -> Dict:
        """Analyze cell morphology based on description"""
        description_lower = description.lower()
        
        # Get relevant indicators for the cell type
        indicators = cell_info.get("severity_indicators", {})
        
        analysis = {
            "cell_type": cell_type,
            "normal_characteristics": cell_info["characteristics"],
            "observed_features": [],
            "abnormalities": [],
            "significance": "normal"
        }
        
        # Check for abnormal features
        for feature_type, possible_values in indicators.items():
            for value in possible_values:
                if value.lower() in description_lower:
                    analysis["observed_features"].append(f"{feature_type}: {value}")
                    if value.lower() != "normal":
                        analysis["abnormalities"].append(f"Abnormal {feature_type}: {value}")
        
        # Assess significance
        if len(analysis["abnormalities"]) > 2:
            analysis["significance"] = "high"
        elif len(analysis["abnormalities"]) > 0:
            analysis["significance"] = "moderate"
        
        return analysis

class BloodCountAnalyzerTool(BaseTool):
    name: str = "blood_count_analyzer"
    description: str = "Analyze blood cell counts and distributions"
    
    def _run(self, cell_type: str, count_data: Dict) -> str:
        """Analyze blood cell counts"""
        cell_lower = cell_type.lower()
        
        if cell_lower in BLOOD_CELL_KNOWLEDGE:
            cell_info = BLOOD_CELL_KNOWLEDGE[cell_lower]
            analysis = self._analyze_count(cell_type, count_data, cell_info)
            return json.dumps(analysis, indent=2)
        
        return {"error": "Unknown cell type"}
    
    def _analyze_count(self, cell_type: str, count_data: Dict, cell_info: Dict) -> Dict:
        """Analyze cell count data"""
        normal_range = cell_info["normal_count"]
        analysis = {
            "cell_type": cell_type,
            "normal_range": normal_range,
            "current_count": count_data.get("count", "unknown"),
            "status": "normal",
            "possible_conditions": []
        }
        
        # Check if count is provided and analyze
        if "count" in count_data:
            count = self._parse_count(count_data["count"])
            normal_low, normal_high = self._parse_range(normal_range)
            
            if count < normal_low:
                analysis["status"] = "low"
                analysis["possible_conditions"] = cell_info["abnormalities"]["low"]
            elif count > normal_high:
                analysis["status"] = "high"
                analysis["possible_conditions"] = cell_info["abnormalities"]["high"]
        
        return analysis
    
    def _parse_count(self, count_str: str) -> float:
        """Parse count value from string"""
        try:
            return float(count_str.replace(",", ""))
        except:
            return 0.0
    
    def _parse_range(self, range_str: str) -> tuple[float, float]:
        """Parse normal range values"""
        try:
            low, high = range_str.split("-")
            return (float(low.replace(",", "")), float(high.replace(",", "")))
        except:
            return (0.0, 0.0)

class BloodHealthRiskAssessorTool(BaseTool):
    name: str = "blood_health_risk_assessor"
    description: str = "Assess blood health risks based on detected cell abnormalities"
    
    def _run(self, cell_type: str, count_data: Dict, abnormalities: List[str] = None) -> str:
        """Assess blood health risks based on cell analysis"""
        risk_factors = []
        
        # Cell count-based risks
        if "count" in count_data:
            count = count_data["count"]
            if cell_type.lower() == "rbc":
                if count < 4000000:  # Low RBC count
                    risk_factors.extend(["Anemia risk", "Oxygen transport deficiency"])
                elif count > 6000000:  # High RBC count
                    risk_factors.extend(["Polycythemia risk", "Blood viscosity issues"])
            elif cell_type.lower() == "wbc":
                if count < 4000:  # Low WBC count
                    risk_factors.extend(["Immunodeficiency risk", "Infection susceptibility"])
                elif count > 11000:  # High WBC count
                    risk_factors.extend(["Infection/inflammation", "Possible leukemia"])
            elif cell_type.lower() == "platelets":
                if count < 150000:  # Low platelet count
                    risk_factors.extend(["Bleeding disorders", "Clotting problems"])
                elif count > 450000:  # High platelet count
                    risk_factors.extend(["Thrombosis risk", "Cardiovascular complications"])
        
        # Morphology-based risks
        if abnormalities:
            for abnormality in abnormalities:
                if "sickle" in abnormality.lower():
                    risk_factors.append("Sickle cell disease")
                elif "spherical" in abnormality.lower():
                    risk_factors.append("Hereditary spherocytosis")
                elif "hypochromic" in abnormality.lower():
                    risk_factors.append("Iron deficiency anemia")
        
        risk_level = "High" if len(risk_factors) > 2 else "Moderate" if len(risk_factors) > 0 else "Low"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": [
                "Complete blood count (CBC) test",
                "Hematologist consultation if abnormal",
                "Regular blood monitoring",
                "Follow-up testing as recommended"
            ]
        }

class BloodCellAIAgent:
    """Main AI Agent for Blood Cell Detection and Analysis"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.api_key = api_key
        print("Initializing LLM...")
        self.llm = self._get_working_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize tools
        self.tools = [
            BloodCellAnalysisTool(),
            CellMorphologyTool(),
            BloodCountAnalyzerTool(),
            BloodHealthRiskAssessorTool()
        ]
        
        # Initialize agent with OpenAI functions agent which better handles multiple tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="structured-chat-zero-shot-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
        if not self.api_key:
            raise ValueError("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")

        models_to_try = ["gemma2-9b-it"]  # Using only gemma2-9b-it model
        
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test")
                    if test_response:
                        return llm
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                return retry_with_exponential_backoff(test_model)
                
            except Exception as e:
                error_msg = str(e).lower()
                if "over capacity" in error_msg or "503" in str(e):
                    continue
                else:
                    # For other errors, try next model
                    continue
        
        # If all models fail, raise an exception
        raise Exception("All GROQ models are currently unavailable")
    
    def analyze_blood_sample(self, 
                          image_description: str,
                          detected_cells: List[str],
                          confidences: List[float],
                          count_data: Dict,
                          morphology: str = "") -> Dict:
        """Comprehensive blood sample analysis with enhanced retry logic"""
        
        # Ensure LLM is initialized
        if self.llm is None:
            try:
                print("Initializing LLM...")
                self._initialize_llm()
                # Verify initialization
                if not self.llm:
                    raise Exception("LLM initialization failed - no model instance created")
                # Test the model
                test_response = self.llm.invoke("Test connection")
                if not test_response:
                    raise Exception("LLM initialization failed - no response from model")
            except Exception as e:
                print(f"LLM initialization failed: {str(e)}")
                print("Please ensure GROQ_API_KEY is set correctly")
                return {
                    "error": "LLM initialization failed",
                    "fallback_analysis": self._generate_fallback_analysis(
                        detected_cells, confidences, count_data
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "status": "fallback"
                }
        
        prompt = f"""
        You are a world-class hematologist AI. Your task is to provide a detailed, professional, and insightful analysis of a blood smear image based on the provided data.

        **Patient Data:**
        - **Image Description:** {image_description}
        - **Morphology Notes:** {morphology}
        - **Count Data:** {json.dumps(count_data, indent=2)}

        **Analysis Request:**
        Generate a comprehensive hematology report. The report must be structured, clear, and provide actionable insights. Follow the structure below precisely.

        **--- COMPREHENSIVE HEMATOLOGY REPORT ---**

        **1. OVERVIEW & SAMPLE QUALITY:**
           - Briefly summarize the overall findings.
           - Comment on the sample quality based on the image description (e.g., "clear, well-stained smear," "presence of artifacts").

        **2. DETAILED CELLULAR ANALYSIS:**

           **A. Red Blood Cells (RBCs):**
              - **Count:** {count_data.get('RBC_count', 'N/A')}
              - **Morphology & Characteristics:** Analyze RBC size (normocytic, microcytic, macrocytic), shape (e.g., biconcave, sickle, spherical), and color (normochromic, hypochromic).
              - **Clinical Significance:** Based on the count and morphology, what are the potential implications? (e.g., "Normal count and morphology suggest no signs of anemia.", "Microcytic, hypochromic cells may indicate iron deficiency anemia.").

           **B. White Blood Cells (WBCs):**
              - **Count:** {count_data.get('WBC_count', 'N/A')}
              - **Morphology & Shape Analysis:** Analyze WBC characteristics. Use the shape analysis data:
                 - **Average Aspect Ratio:** {count_data.get('shape_analysis', {{}}).get('wbc', {{}}).get('avg_aspect_ratio', 'N/A'):.2f} (Note: 1.0 is perfectly round).
                 - **Average Solidity:** {count_data.get('shape_analysis', {{}}).get('wbc', {{}}).get('avg_solidity', 'N/A'):.2f} (Note: 1.0 is a perfect convex shape).
              - **Clinical Significance:** Interpret the WBC count and morphology. (e.g., "Elevated WBC count with immature forms may suggest infection or a leukemoid reaction.", "Normal count and morphology.").

           **C. Platelets:**
              - **Count:** {count_data.get('Platelet_count', 'N/A')}
              - **Morphology & Distribution:** Analyze platelet size, shape, and whether they appear clumped or evenly distributed.
                 - **Average Aspect Ratio:** {count_data.get('shape_analysis', {{}}).get('platelets', {{}}).get('avg_aspect_ratio', 'N/A'):.2f}
              - **Clinical Significance:** What does the platelet count and appearance suggest? (e.g., "Low platelet count (thrombocytopenia) increases risk of bleeding.", "Platelet clumping can be an artifact or indicate activation.").

        **3. KEY FINDINGS & ABNORMALITIES:**
           - Bullet-point list of the most critical findings from the analysis.
           - Example: "- Marked anisopoikilocytosis of RBCs.", "- Presence of atypical lymphocytes."

        **4. POTENTIAL CLINICAL IMPLICATIONS & RISK ASSESSMENT:**
           - Synthesize the findings to assess potential health risks.
           - Discuss possible conditions or diseases indicated by the results (e.g., anemia, infection, clotting disorders).

        **5. RECOMMENDATIONS:**
           - **Further Tests:** Suggest specific follow-up tests (e.g., "Complete Blood Count (CBC) with differential," "Iron studies," "Bone marrow biopsy if abnormalities are severe.").
           - **Clinical Actions:** Recommend actions for the healthcare provider (e.g., "Monitor patient for signs of infection," "Referral to a hematologist for further evaluation.").

        **--- END OF REPORT ---**
        """
        
        def execute_analysis():
            # Force LLM reinitialization
            try:
                self.llm = self._get_working_llm()
                
                # Generate a shorter, focused prompt first to test the LLM
                test_prompt = "Provide a brief analysis of blood cell counts."
                test_response = self.llm.invoke(test_prompt)
                
                if not test_response:
                    raise Exception("LLM failed initial test")
                
                # If test passes, proceed with full analysis
                response = self.llm.invoke(prompt)
                
                if not response or len(str(response).strip()) < 100:
                    raise Exception("Generated analysis is too short")
                
                return response
                
            except Exception as e:
                print(f"Analysis execution error: {str(e)}")
                raise e

        # Enhanced retry mechanism with progressive backoff
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"Attempting report generation (attempt {attempt + 1}/{max_attempts})")
                
                response = retry_with_exponential_backoff(
                    execute_analysis,
                    max_retries=2,
                    base_delay=5,
                    max_delay=20
                )
                
                # Validate response
                if response and len(str(response).strip()) > 200:  # Ensure meaningful content
                    return {
                        "analysis": response,
                        "timestamp": datetime.now().isoformat(),
                        "agent_version": "1.2",
                        "status": "success",
                        "attempt": attempt + 1
                    }
                else:
                    raise Exception("Invalid response format")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print(f"Waiting before next attempt...")
                    time.sleep(10)  # Wait between major attempts
                continue
                
        # If all attempts fail, return detailed fallback
        print("All report generation attempts failed, using fallback analysis")
        return {
            "error": "Report generation failed after multiple attempts",
            "fallback_analysis": self._generate_fallback_analysis(
                detected_cells, confidences, count_data
            ),
            "timestamp": datetime.now().isoformat(),
            "status": "fallback",
            "attempts_made": max_attempts
        }
    
    def _generate_fallback_analysis(self, cells: List[str], confidences: List[float], count_data: Dict) -> str:
        """Generate detailed fallback analysis when AI agent fails"""
        try:
            # Calculate basic statistics
            avg_confidence = sum(confidences)/len(confidences) if confidences else 0
            cell_counts = {}
            for cell in cells:
                cell_counts[cell] = cell_counts.get(cell, 0) + 1

            # Generate detailed fallback report
            report = f"""
            **AUTOMATED BLOOD SAMPLE ANALYSIS REPORT**
            
            **1. DETECTION SUMMARY:**
            - Total Cells Detected: {len(cells)}
            - Detection Confidence: {avg_confidence:.2f}
            - Cell Type Distribution: {', '.join(f'{k}: {v}' for k, v in cell_counts.items())}
            
            **2. QUANTITATIVE ANALYSIS:**
            """
            
            # Add count data if available
            if count_data:
                report += "\nCell Counts:\n"
                for cell_type, count in count_data.items():
                    if isinstance(count, (int, float)):
                        report += f"- {cell_type}: {count:,}\n"

                # Add shape analysis if available
                if 'shape_analysis' in count_data:
                    report += "\nMorphological Indicators:\n"
                    for cell_type, metrics in count_data['shape_analysis'].items():
                        report += f"- {cell_type.upper()}:\n"
                        for metric, value in metrics.items():
                            report += f"  • {metric}: {value:.3f}\n"

            report += """
            **3. RECOMMENDATIONS:**
            1. Conduct manual verification of cell counts
            2. Perform detailed morphological examination
            3. Consider additional confirmatory tests
            4. Schedule follow-up with hematology specialist
            5. Document any clinical symptoms for correlation
            
            **4. LIMITATIONS:**
            - This is an automated preliminary analysis
            - AI-assisted report generation was unavailable
            - Professional laboratory verification is required
            - Results should be interpreted in clinical context
            
            **5. NEXT STEPS:**
            1. Complete blood count (CBC) with differential
            2. Manual microscopic examination
            3. Clinical correlation with patient symptoms
            4. Consider specialized hematological tests based on findings
            
            **Note:** This is a fallback analysis generated due to AI service unavailability. 
            Please ensure professional medical interpretation of these results.
            """
            
            return report
            
        except Exception as e:
            # Ultimate fallback if even basic analysis fails
            return f"""
            **Basic Analysis Summary**
            
            Detected Cells: {', '.join(cells) if cells else 'None'}
            Status: Limited analysis available
            
            Note: Full analysis unavailable. Please consult laboratory professional.
            Error details: {str(e)}
            """

class HematologyResearchAgent:
    """Research Assistant for Hematology Literature"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM with retries"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.llm = self._get_working_llm()
                return
            except Exception as e:
                print(f"LLM initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                else:
                    raise
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
        if not self.api_key:
            raise Exception("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")

        models_to_try = [
            "llama3-70b-8192",  # Primary model
            "mixtral-8x7b-32768",  # First fallback
            "gemma2-9b-it",  # Second fallback
            "llama3-8b-8192"  # Last resort
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key,
                        request_timeout=30  # Add timeout
                    )
                    # Test with a simple system-check prompt
                    test_response = llm.invoke("Perform a simple system check.")
                    if test_response and len(str(test_response).strip()) > 0:
                        print(f"Successfully initialized model: {model_name}")
                        return llm
                    raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                return retry_with_exponential_backoff(test_model, max_retries=3, base_delay=2)
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                print(f"Error with model {model_name}: {error_msg}")
                
                if any(msg in error_msg for msg in ["over capacity", "503", "timeout", "network"]):
                    continue  # Try next model for capacity/network issues
                elif "api key" in error_msg:
                    raise Exception("Invalid GROQ API key") from e
                else:
                    continue  # Try next model for other errors
        
        # If all models fail, provide detailed error
        error_msg = f"All GROQ models are currently unavailable. Last error: {str(last_error)}"
        print(error_msg)  # Log the error
        raise Exception(error_msg)
    
    def search_hematology_literature(self, condition: str) -> str:
        """Search hematology literature for blood conditions"""
        prompt = f"""
        Provide recent hematology research findings about: {condition}
        
        Include:
        1. Latest diagnostic approaches
        2. Treatment protocols
        3. Risk factors and complications
        4. Management practices
        5. Laboratory reference ranges
        """
        
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Unable to search literature: {str(e)}"

class BloodDataAnalysisAgent:
    """Data Analysis Agent for Blood Health Trends"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_blood_health_trends(self, blood_history: List[Dict]) -> Dict:
        """Analyze blood health trends from history"""
        if not blood_history:
            return {"message": "No blood analysis history available"}
        
        # Analyze trends
        cell_types = [entry.get("cell_type") for entry in blood_history]
        counts = [entry.get("count", 0) for entry in blood_history]
        confidences = [entry.get("confidence", 0) for entry in blood_history]
        
        trend_analysis = {
            "total_analyses": len(blood_history),
            "most_analyzed_cell": max(set(cell_types), key=cell_types.count) if cell_types else None,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "count_trend": "Increasing" if len(counts) > 1 and counts[-1] > counts[0] else "Stable",
            "recommendations": self._generate_blood_trend_recommendations(cell_types, counts, confidences)
        }
        
        return trend_analysis
    
    def _generate_blood_trend_recommendations(self, cell_types: List[str], counts: List[float], confidences: List[float]) -> List[str]:
        """Generate recommendations based on blood analysis trends"""
        recommendations = []
        
        if len(cell_types) > 1:
            if cell_types[-1] == cell_types[-2]:
                recommendations.append("Consistent cell type analysis - consider comprehensive blood panel")
            
            if confidences[-1] < 0.7:
                recommendations.append("Low confidence in recent analysis - recommend manual verification")
        
        # Check for concerning trends
        if len(counts) > 1:
            if counts[-1] < counts[0] * 0.8:  # 20% decrease
                recommendations.append("Declining cell count trend - consult hematologist")
            elif counts[-1] > counts[0] * 1.2:  # 20% increase
                recommendations.append("Increasing cell count trend - monitor closely")
        
        recommendations.extend([
            "Continue regular blood monitoring",
            "Maintain detailed health records",
            "Follow up with healthcare provider as recommended"
        ])
        
        return recommendations

# Utility functions
def create_agent_instance(agent_type: str, api_key: str):
    """Create agent instance based on type"""
    if agent_type == "blood":
        return BloodCellAIAgent(api_key)
    elif agent_type == "research":
        return HematologyResearchAgent(api_key)
    elif agent_type == "data":
        return BloodDataAnalysisAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_recommendations(cell_type: str, count_data: Dict) -> Dict:
    """Get agent recommendations for blood cell analysis"""
    recommendations = {
        "immediate_actions": [],
        "short_term": [],
        "long_term": [],
        "monitoring": []
    }
    
    # Cell-specific recommendations
    cell_lower = cell_type.lower()
    
    if cell_lower in BLOOD_CELL_KNOWLEDGE:
        cell_info = BLOOD_CELL_KNOWLEDGE[cell_lower]
        
        # Check for count abnormalities
        if "count" in count_data:
            try:
                count = float(str(count_data["count"]).replace(",", ""))
                normal_range = cell_info["normal_count"]
                low, high = map(float, normal_range.split("-")[0].replace(",", "").split())
                
                if count < low:
                    recommendations["immediate_actions"].extend([
                        f"Verify low {cell_type} count",
                        "Consider complete blood count (CBC) test"
                    ])
                    recommendations["short_term"].extend([
                        "Monitor for signs of underlying conditions",
                        f"Check for {', '.join(cell_info['abnormalities']['low'])}"
                    ])
                elif count > high:
                    recommendations["immediate_actions"].extend([
                        f"Verify high {cell_type} count",
                        "Schedule follow-up blood tests"
                    ])
                    recommendations["short_term"].extend([
                        "Monitor for associated symptoms",
                        f"Check for {', '.join(cell_info['abnormalities']['high'])}"
                    ])
            except:
                recommendations["immediate_actions"].append("Verify cell count measurement")
    
    # General monitoring recommendations
    recommendations["monitoring"].extend([
        "Regular complete blood count (CBC)",
        "Track cell count trends",
        "Monitor cell morphology",
        "Document any symptoms"
    ])
    
    # Long-term recommendations
    recommendations["long_term"].extend([
        "Establish baseline blood values",
        "Regular health check-ups",
        "Maintain medical records"
    ])
    
    return recommendations
