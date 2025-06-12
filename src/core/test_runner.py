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
        """
        Extract letter answer from model response with enhanced support for thinking models.

        This method handles various response patterns including:
        - Direct answers: "B", "The answer is B"
        - Thinking model responses with reasoning before the answer
        - Various answer formats and phrasings
        """
        # Clean the response
        response = response.strip()
        original_response = response
        response_upper = response.upper()

        # Common answer patterns to look for (case-insensitive)
        answer_patterns = [
            # Final answer patterns (highest priority)
            r'(?:FINAL|MY|THE)\s+ANSWER\s*[:=]?\s*\**\s*([A-D])\b',
            r'ANSWER\s*[:=]?\s*\**\s*([A-D])\b',

            # "Is" patterns
            r'(?:CORRECT\s+)?ANSWER\s+IS\s*[:=]?\s*\**\s*([A-D])\b',
            r'(?:THE\s+)?(?:BEST|RIGHT|CORRECT)\s+(?:ANSWER|OPTION|CHOICE)\s+IS\s*[:=]?\s*\**\s*([A-D])\b',

            # Choice/Option patterns
            r'(?:I\s+)?(?:CHOOSE|SELECT|PICK)\s*[:=]?\s*(?:OPTION\s+)?([A-D])\b',
            r'OPTION\s+([A-D])\s+IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|BEST)',

            # Therefore/Thus patterns (often at the end of reasoning)
            r'(?:THEREFORE|THUS|HENCE|SO),?\s*(?:THE\s+)?(?:ANSWER\s+)?(?:IS\s+)?([A-D])\b',

            # Conclusion patterns
            r'(?:IN\s+)?CONCLUSION,?\s*(?:THE\s+)?(?:ANSWER\s+)?(?:IS\s+)?([A-D])\b',

            # Direct statement patterns
            r'^([A-D])\s*(?:\.|$)',  # Just the letter at the start
            r'([A-D])\s+IS\s+(?:THE\s+)?(?:CORRECT|RIGHT)',
            r'\b([A-D])\s*\.\s*$',  # Letter followed by period at the end
        ]

        # Try each pattern, starting with the most specific
        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, response_upper))
            if matches:
                # For thinking models, prefer the LAST match (final answer)
                return matches[-1].group(1)

        # If no pattern matched, look for the last standalone letter that appears to be an answer
        # This handles cases where the model might discuss options but then state the answer simply

        # Split response into sentences/lines
        lines = response.split('\n')

        # Check last few lines first (where answer is most likely to be)
        for line in reversed(lines[-3:]):
            line_upper = line.strip().upper()
            if line_upper and len(line_upper) < 50:  # Short lines are more likely to be answers
                # Check if this line contains a standalone letter
                if re.match(r'^[A-D]\.?$', line_upper):
                    return line_upper[0]
                # Check for simple patterns in short lines
                match = re.search(r'\b([A-D])\b(?:\.|$)', line_upper)
                if match:
                    return match.group(1)

        # Last resort: find all occurrences of A-D and use heuristics
        all_letters = re.findall(r'\b([A-D])\b', response_upper)

        if all_letters:
            # Count occurrences of each letter
            letter_counts = {}
            for letter in all_letters:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1

            # If one letter appears significantly more than others, it might be emphasized as the answer
            max_count = max(letter_counts.values())
            if max_count >= 2:
                for letter, count in letter_counts.items():
                    if count == max_count:
                        # Check if this letter appears near answer-related words
                        answer_keywords = ['ANSWER', 'CORRECT', 'CHOICE', 'OPTION', 'BEST', 'RIGHT']
                        for keyword in answer_keywords:
                            if keyword in response_upper:
                                # Find positions of the keyword and the letter
                                keyword_pos = response_upper.rfind(keyword)
                                letter_positions = [m.start() for m in re.finditer(f'\\b{letter}\\b', response_upper)]
                                # Check if the letter appears near (within 50 chars) the keyword
                                for pos in letter_positions:
                                    if abs(pos - keyword_pos) < 50:
                                        return letter

            # If no clear answer, return the last mentioned letter
            return all_letters[-1]

        # Default to empty if no valid answer found
        logger.warning(f"Could not extract answer from response: {original_response[:100]}...")
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