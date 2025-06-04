import pandas as pd
import os
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate exam data from CSV files"""

    @staticmethod
    def load_questions(file_path: str) -> Optional[pd.DataFrame]:
        """Load questions from CSV file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Questions file not found: {file_path}")
                return None

            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ['id', 'question', 'option_a', 'option_b', 'option_c', 'option_d']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Expected: {required_columns}")
                return None

            # Check for empty values
            if df[required_columns].isnull().any().any():
                logger.error("Found empty values in questions data")
                return None

            logger.info(f"Loaded {len(df)} questions successfully")
            return df

        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return None

    @staticmethod
    def load_answers(file_path: str) -> Optional[pd.DataFrame]:
        """Load answers from CSV file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Answers file not found: {file_path}")
                return None

            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ['id', 'correct_answer']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Expected: {required_columns}")
                return None

            # Validate answer format (should be A, B, C, or D)
            valid_answers = ['A', 'B', 'C', 'D']
            if not df['correct_answer'].isin(valid_answers).all():
                logger.error("Invalid answer format. Answers must be A, B, C, or D")
                return None

            logger.info(f"Loaded {len(df)} answers successfully")
            return df

        except Exception as e:
            logger.error(f"Error loading answers: {str(e)}")
            return None

    @staticmethod
    def validate_data(questions_df: pd.DataFrame, answers_df: pd.DataFrame) -> bool:
        """Validate that questions and answers match"""
        try:
            question_ids = set(questions_df['id'].values)
            answer_ids = set(answers_df['id'].values)

            if question_ids != answer_ids:
                missing_answers = question_ids - answer_ids
                extra_answers = answer_ids - question_ids

                if missing_answers:
                    logger.error(f"Missing answers for questions: {missing_answers}")
                if extra_answers:
                    logger.error(f"Extra answers without questions: {extra_answers}")

                return False

            return True

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False