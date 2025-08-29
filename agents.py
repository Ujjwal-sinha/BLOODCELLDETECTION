"""
AI Agent for Blood Cell Analysis - BloodCellAI
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from utils import retry_with_exponential_backoff, generate_fallback_response

class BloodCellAgent:
    """
    AI Agent for analyzing blood cell detection results
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.models = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        self.llm = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the language model"""
        for model_name in self.models:
            try:
                def init_model():
                    self.llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key
                    )
                    test_response = self.llm.invoke("Test blood cell analysis")
                    if test_response:
                        print(f"Initialized model: {model_name}")
                        return True
                    return False
                result = retry_with_exponential_backoff(init_model)
                if result:
                    return
            except Exception as e:
                print(f"Failed to initialize {model_name}: {str(e)}")
                continue
        print("All models failed to initialize. Using fallback responses.")
    
    def analyze_blood_cells(self, image_description: str, predictions: dict) -> str:
        """
        Analyze blood cell detection results
        """
        if not self.llm:
            detected_cell = predictions['detections'][0]['class'] if predictions and predictions['detections'] else "Unknown"
            return generate_fallback_response(detected_cell, image_description)
        
        try:
            cell_counts = {'Platelets': 0, 'RBC': 0, 'WBC': 0}
            if predictions and predictions['detections']:
                for det in predictions['detections']:
                    cell_counts[det['class']] += 1
            
            prompt = f"""
            You are an expert hematologist analyzing a blood cell image.
            
            Image Description: {image_description}
            Detection Results:
            - Platelets: {cell_counts['Platelets']}
            - RBC: {cell_counts['RBC']}
            - WBC: {cell_counts['WBC']}
            
            Provide a detailed analysis including:
            1. Cell type identification and counts
            2. Potential clinical implications
            3. Recommendations for further testing
            """
            
            def query_model():
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            
            analysis = retry_with_exponential_backoff(query_model)
            if analysis:
                return analysis
            return generate_fallback_response("Unknown", image_description)
        except Exception as e:
            print(f"Error in analysis: {e}")
            return generate_fallback_response("Unknown", image_description)