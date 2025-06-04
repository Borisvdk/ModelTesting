import json
import time
import requests
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaModel:
    """Interface for Ollama API"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_tags = f"{base_url}/api/tags"

    def initialize(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(self.api_tags)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                if self.model_name in available_models:
                    logger.info(f"Model {self.model_name} is available")
                    return True
                else:
                    logger.error(f"Model {self.model_name} not found. Available: {available_models}")
                    return False
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama connection error: {str(e)}")
            return False

    def generate_response(self, prompt: str, system_prompt: str = "") -> Dict[str, any]:
        """Generate response from model"""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for deterministic answers
                    "top_p": 0.9,
                    "seed": 42
                }
            }

            start_time = time.time()
            response = requests.post(self.api_generate, json=payload)
            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "response_time": response_time,
                    "success": True
                }
            else:
                return {
                    "response": "",
                    "response_time": response_time,
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }

        except Exception as e:
            return {
                "response": "",
                "response_time": 0,
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []