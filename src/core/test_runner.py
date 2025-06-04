import re
import time
import yaml
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import os

from ..models import OllamaModel
from .data_loader import DataLoader
from .results_manager import ResultsManager

logger = logging.getLogger(__name__)


class TestRunner:
    """Run tests with LLM models"""

    def __init__(self, config_path: str = "config/prompts.yaml"):
        self.config_path = config_path
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from config"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            # Return default prompts
            return {
                "system_prompt": "Answer with only A, B, C, or D.",
                "question_template": "Question: {question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}"
            }

    def extract_letter(self, response: str) -> str:
        """Extract letter answer from model response"""
        # Clean the response
        response = response.strip().upper()

        # Try to find a single letter A-D
        match = re.search(r'^[A-D]', response)
        if match:
            return match.group(0)

        # Try to find letter anywhere in response
        match = re.search(r'[A-D]', response)
        if match:
            return match.group(0)

        # Default to empty if no valid answer found
        return ""

    def format_question_prompt(self, question_row: Dict) -> str:
        """Format question using template"""
        return self.prompts['question_template'].format(
            question=question_row['question'],
            option_a=question_row['option_a'],
            option_b=question_row['option_b'],
            option_c=question_row['option_c'],
            option_d=question_row['option_d']
        )

    def run_test(self, model_name: str, questions_df, answers_df,
                 progress_callback=None) -> Tuple[List[Dict], float]:
        """Run complete test with model"""
        # Initialize model
        model = OllamaModel(model_name)
        if not model.initialize():
            raise Exception(f"Failed to initialize model: {model_name}")

        # Prepare results container
        results = []
        total_questions = len(questions_df)

        # Run through each question
        for idx, (_, question) in enumerate(questions_df.iterrows()):
            # Update progress
            if progress_callback:
                progress_callback(idx / total_questions, f"Processing question {idx + 1}/{total_questions}")

            # Format prompt
            prompt = self.format_question_prompt(question)

            # Get response
            response_data = model.generate_response(prompt, self.prompts['system_prompt'])

            if response_data['success']:
                # Parse answer
                answer = self.extract_letter(response_data['response'])

                # Get correct answer
                correct_answer = answers_df[answers_df['id'] == question['id']]['correct_answer'].iloc[0]
                is_correct = answer == correct_answer

                # Store result
                results.append({
                    'question_id': question['id'],
                    'question': question['question'],
                    'model_response': response_data['response'],
                    'extracted_answer': answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'response_time': response_data['response_time']
                })
            else:
                # Handle failed response
                correct_answer = answers_df[answers_df['id'] == question['id']]['correct_answer'].iloc[0]
                results.append({
                    'question_id': question['id'],
                    'question': question['question'],
                    'model_response': "ERROR: " + response_data.get('error', 'Unknown error'),
                    'extracted_answer': "",
                    'correct_answer': correct_answer,
                    'is_correct': False,
                    'response_time': 0
                })

        # Calculate score
        correct_count = sum(r['is_correct'] for r in results)
        score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0

        # Save results
        test_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name.replace(':', '_')}"
        ResultsManager.save_test_results(test_id, model_name, results, score_percentage)

        return results, score_percentage