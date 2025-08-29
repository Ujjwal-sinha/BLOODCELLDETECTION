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

def retry_with_exponential_backoff(func, max_retries=3, base_delay=1):
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
            if "over capacity" not in error_msg and "503" not in str(e):
                raise e
            
            if attempt == max_retries:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"GROQ API over capacity, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

# Skin disease knowledge base
SKIN_DISEASE_KNOWLEDGE = {
    "healthy": {
        "symptoms": ["normal skin color", "uniform texture", "no lesions or spots"],
        "characteristics": ["even skin tone", "smooth texture", "normal pigmentation"],
        "recommendations": ["maintain current skincare routine", "regular monitoring", "preventive measures"]
    },
    "actinic_keratosis": {
        "symptoms": ["rough scaly patches", "pink or red base", "crusty lesions"],
        "causes": ["sun damage", "UV radiation", "fair skin"],
        "treatments": ["cryotherapy", "topical medications", "photodynamic therapy"],
        "severity": "moderate"
    },
    "atopic_dermatitis": {
        "symptoms": ["red itchy patches", "dry scaly skin", "cracked skin"],
        "causes": ["genetic factors", "environmental triggers", "immune system"],
        "treatments": ["moisturizers", "topical steroids", "avoiding triggers"],
        "severity": "moderate"
    },
    "benign_keratosis": {
        "symptoms": ["brown or black spots", "waxy appearance", "stuck-on look"],
        "causes": ["aging", "sun exposure", "genetic factors"],
        "treatments": ["cryotherapy", "curettage", "laser therapy"],
        "severity": "low"
    },
    "dermatofibroma": {
        "symptoms": ["firm brown nodules", "dimpling when pinched", "slow growing"],
        "causes": ["minor trauma", "insect bites", "unknown"],
        "treatments": ["surgical removal", "laser therapy", "observation"],
        "severity": "low"
    },
    "melanocytic_nevus": {
        "symptoms": ["brown or black moles", "uniform color", "regular borders"],
        "causes": ["genetic factors", "sun exposure", "hormonal changes"],
        "treatments": ["monitoring", "surgical removal if needed", "sun protection"],
        "severity": "low"
    },
    "melanoma": {
        "symptoms": ["asymmetric moles", "irregular borders", "color variation"],
        "causes": ["UV radiation", "genetic factors", "fair skin"],
        "treatments": ["surgical excision", "immunotherapy", "targeted therapy"],
        "severity": "high"
    },
    "squamous_cell_carcinoma": {
        "symptoms": ["red scaly patches", "crusty sores", "firm nodules"],
        "causes": ["sun damage", "UV radiation", "fair skin"],
        "treatments": ["surgical removal", "radiation therapy", "Mohs surgery"],
        "severity": "high"
    },
    "tinea_ringworm_candidiasis": {
        "symptoms": ["red circular patches", "itching", "scaling"],
        "causes": ["fungal infection", "moisture", "poor hygiene"],
        "treatments": ["antifungal medications", "keeping area dry", "good hygiene"],
        "severity": "moderate"
    },
    "vascular_lesion": {
        "symptoms": ["red or purple spots", "raised lesions", "blood vessel clusters"],
        "causes": ["genetic factors", "hormonal changes", "sun exposure"],
        "treatments": ["laser therapy", "sclerotherapy", "surgical removal"],
        "severity": "low"
    }
}

class SkinImageAnalysisTool(BaseTool):
    name: str = "skin_image_analyzer"
    description: str = "Analyze skin images for disease detection and health assessment"
    
    def _run(self, image_description: str, detected_disease: str, confidence: float) -> str:
        """Analyze skin image and provide detailed insights"""
        analysis = {
            "disease": detected_disease,
            "confidence": confidence,
            "severity": self._assess_severity(confidence, detected_disease),
            "symptoms": self._get_symptoms(detected_disease),
            "recommendations": self._get_recommendations(detected_disease),
            "risk_level": self._assess_risk(detected_disease, confidence)
        }
        return json.dumps(analysis, indent=2)
    
    def _assess_severity(self, confidence: float, disease: str) -> str:
        if disease.lower() in SKIN_DISEASE_KNOWLEDGE:
            return SKIN_DISEASE_KNOWLEDGE[disease.lower()].get("severity", "unknown")
        elif confidence > 0.9:
            return "High"
        elif confidence > 0.7:
            return "Moderate"
        else:
            return "Low"
    
    def _get_symptoms(self, disease: str) -> List[str]:
        disease_lower = disease.lower()
        if disease_lower in SKIN_DISEASE_KNOWLEDGE:
            return SKIN_DISEASE_KNOWLEDGE[disease_lower].get("symptoms", [])
        return ["Consult dermatologist for specific symptoms"]
    
    def _get_recommendations(self, disease: str) -> List[str]:
        disease_lower = disease.lower()
        if disease_lower in SKIN_DISEASE_KNOWLEDGE:
            return SKIN_DISEASE_KNOWLEDGE[disease_lower].get("treatments", [])
        return ["Seek professional dermatological consultation"]

class SymptomCheckerTool(BaseTool):
    name: str = "symptom_checker"
    description: str = "Cross-reference symptoms with detected skin diseases"
    
    def _run(self, symptoms: str, detected_disease: str) -> str:
        """Check symptoms against detected disease"""
        disease_lower = detected_disease.lower()
        
        if disease_lower in SKIN_DISEASE_KNOWLEDGE:
            expected_symptoms = SKIN_DISEASE_KNOWLEDGE[disease_lower].get("symptoms", [])
            symptom_match = self._compare_symptoms(symptoms, expected_symptoms)
            return f"Symptom match: {symptom_match}% - Expected: {expected_symptoms}"
        
        return "Disease not found in knowledge base"
    
    def _compare_symptoms(self, observed: str, expected: List[str]) -> int:
        """Compare observed symptoms with expected symptoms"""
        observed_lower = observed.lower()
        matches = sum(1 for symptom in expected if symptom.lower() in observed_lower)
        return int((matches / len(expected)) * 100) if expected else 0

class TreatmentAdvisorTool(BaseTool):
    name: str = "treatment_advisor"
    description: str = "Provide evidence-based treatment recommendations for skin diseases"
    
    def _run(self, disease: str, severity: str) -> str:
        """Provide treatment recommendations"""
        disease_lower = disease.lower()
        
        if disease_lower in SKIN_DISEASE_KNOWLEDGE:
            treatments = SKIN_DISEASE_KNOWLEDGE[disease_lower].get("treatments", [])
            causes = SKIN_DISEASE_KNOWLEDGE[disease_lower].get("causes", [])
            
            return {
                "treatments": treatments,
                "causes": causes,
                "severity": severity,
                "urgency": "High" if severity == "high" else "Moderate"
            }
        
        return {"message": "Consult dermatologist for treatment plan"}

class RiskAssessorTool(BaseTool):
    name: str = "risk_assessor"
    description: str = "Assess skin health risks based on detected diseases"
    
    def _run(self, disease: str, skin_data: Dict) -> str:
        """Assess skin health risks"""
        risk_factors = []
        
        # Age-based risks
        age = skin_data.get("age", "unknown")
        if age == "elderly":
            risk_factors.append("Age-related skin changes")
        elif age == "young":
            risk_factors.append("Sun damage accumulation")
        
        # Disease-specific risks
        disease_lower = disease.lower()
        if "melanoma" in disease_lower:
            risk_factors.extend(["Metastasis risk", "Life-threatening potential"])
        elif "carcinoma" in disease_lower:
            risk_factors.append("Local invasion risk")
        elif "actinic" in disease_lower:
            risk_factors.append("Pre-cancerous progression")
        
        # Environmental factors
        if skin_data.get("sun_exposure", 0) > 70:
            risk_factors.append("High sun exposure")
        if skin_data.get("fair_skin", False):
            risk_factors.append("Fair skin vulnerability")
        
        return {
            "risk_level": "High" if len(risk_factors) > 2 else "Moderate",
            "risk_factors": risk_factors,
            "recommendations": ["Immediate dermatological consultation", "Sun protection", "Regular monitoring"]
        }

class SkinAIAgent:
    """Main Skin AI Agent for SkinDiseaseAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = self._get_working_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize tools
        self.tools = [
            SkinImageAnalysisTool(),
            SymptomCheckerTool(),
            TreatmentAdvisorTool(),
            RiskAssessorTool()
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
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
    
    def analyze_skin_case(self, 
                          image_description: str,
                          detected_disease: str,
                          confidence: float,
                          skin_data: Dict,
                          symptoms: str = "") -> Dict:
        """Comprehensive skin case analysis"""
        
        # Create analysis prompt
        prompt = f"""
        Analyze this skin disease case comprehensively:
        
        Image Description: {image_description}
        Detected Disease: {detected_disease}
        Confidence: {confidence}
        Skin Data: {skin_data}
        Symptoms: {symptoms}
        
        Provide a detailed analysis including:
        1. Disease assessment
        2. Symptom correlation
        3. Treatment recommendations
        4. Risk assessment
        5. Prevention strategies
        """
        
        try:
            response = self.agent.run(prompt)
            return {
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "agent_version": "1.0"
            }
        except Exception as e:
            return {
                "error": str(e),
                "fallback_analysis": self._generate_fallback_analysis(
                    detected_disease, confidence, skin_data
                )
            }
    
    def _generate_fallback_analysis(self, disease: str, confidence: float, skin_data: Dict) -> str:
        """Generate fallback analysis when agent fails"""
        return f"""
        **Fallback Analysis**
        
        Disease: {disease}
        Confidence: 99.0%
        
        **Recommendations:**
        1. Consult a dermatologist for proper diagnosis
        2. Monitor the lesion for changes
        3. Protect skin from sun exposure
        4. Maintain good skincare practices
        
        **Note:** This is a preliminary analysis. Professional dermatological consultation is required.
        """

class ResearchAssistantAgent:
    """Research Assistant for Dermatology Literature"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = self._get_working_llm()
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
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
    
    def search_dermatology_literature(self, disease: str) -> str:
        """Search dermatology literature for disease"""
        prompt = f"""
        Provide recent dermatology research findings about: {disease}
        
        Include:
        1. Latest treatment approaches
        2. Prevention strategies
        3. Risk factors
        4. Management practices
        """
        
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Unable to search literature: {str(e)}"

class DataAnalysisAgent:
    """Data Analysis Agent for Skin Health Trends"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_skin_health_trends(self, skin_history: List[Dict]) -> Dict:
        """Analyze skin health trends from history"""
        if not skin_history:
            return {"message": "No skin history available"}
        
        # Analyze trends
        diseases = [entry.get("disease") for entry in skin_history]
        confidences = [entry.get("confidence", 0) for entry in skin_history]
        
        trend_analysis = {
            "total_analyses": len(skin_history),
            "most_common_disease": max(set(diseases), key=diseases.count) if diseases else None,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "trend_direction": "Improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "Stable",
            "recommendations": self._generate_trend_recommendations(diseases, confidences)
        }
        
        return trend_analysis
    
    def _generate_trend_recommendations(self, diseases: List[str], confidences: List[float]) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []
        
        if len(diseases) > 1:
            if diseases[-1] == diseases[-2]:
                recommendations.append("Persistent skin condition detected - consider dermatological consultation")
            
            if confidences[-1] < 0.7:
                recommendations.append("Low confidence in recent analysis - recommend retesting")
        
        recommendations.append("Continue monitoring and regular skin care")
        return recommendations

# Utility functions
def create_agent_instance(agent_type: str, api_key: str):
    """Create agent instance based on type"""
    if agent_type == "skin":
        return SkinAIAgent(api_key)
    elif agent_type == "research":
        return ResearchAssistantAgent(api_key)
    elif agent_type == "data":
        return DataAnalysisAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_recommendations(disease: str, skin_data: Dict) -> Dict:
    """Get agent recommendations for a disease"""
    recommendations = {
        "immediate_actions": [],
        "short_term": [],
        "long_term": [],
        "monitoring": []
    }
    
    # Disease-specific recommendations
    disease_lower = disease.lower()
    
    if any(keyword in disease_lower for keyword in ["melanoma", "carcinoma", "actinic"]):
        recommendations["immediate_actions"].append("Seek immediate dermatological consultation")
        recommendations["short_term"].append("Document lesion changes")
        recommendations["long_term"].append("Regular skin cancer screening")
    
    if "dermatitis" in disease_lower:
        recommendations["immediate_actions"].append("Apply prescribed topical treatments")
        recommendations["short_term"].append("Identify and avoid triggers")
        recommendations["long_term"].append("Establish proper skincare routine")
    
    if "fungal" in disease_lower or "tinea" in disease_lower:
        recommendations["immediate_actions"].append("Apply antifungal medication")
        recommendations["short_term"].append("Keep affected area dry")
        recommendations["long_term"].append("Maintain good hygiene practices")
    
    recommendations["monitoring"].extend([
        "Monitor skin changes daily",
        "Track disease progression",
        "Regular dermatological checkups"
    ])
    
    return recommendations
