import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manage test results and scores"""

    RESULTS_DIR = "data/results"
    RESPONSES_DIR = os.path.join(RESULTS_DIR, "responses")
    SCORES_FILE = os.path.join(RESULTS_DIR, "scores.csv")

    @classmethod
    def ensure_directories(cls):
        """Ensure results directories exist"""
        os.makedirs(cls.RESPONSES_DIR, exist_ok=True)

    @classmethod
    def save_test_results(cls, test_id: str, model_name: str,
                          results: List[Dict], score_percentage: float):
        """Save individual test results"""
        cls.ensure_directories()

        # Save detailed results as JSON
        results_file = os.path.join(cls.RESPONSES_DIR, f"{test_id}.json")
        test_data = {
            'test_id': test_id,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_questions': len(results),
                'correct_answers': sum(r['is_correct'] for r in results),
                'score_percentage': score_percentage,
                'avg_response_time': sum(r['response_time'] for r in results) / len(results) if results else 0
            }
        }

        with open(results_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        # Update master scores CSV
        cls._update_scores_csv(test_data)

        logger.info(f"Saved test results: {test_id}")

    @classmethod
    def _update_scores_csv(cls, test_data: Dict):
        """Update master scores CSV"""
        # Create new row
        new_row = {
            'test_id': test_data['test_id'],
            'model': test_data['model'],
            'timestamp': test_data['timestamp'],
            'total_questions': test_data['summary']['total_questions'],
            'correct_answers': test_data['summary']['correct_answers'],
            'score_percentage': test_data['summary']['score_percentage'],
            'avg_response_time': test_data['summary']['avg_response_time']
        }

        # Load existing scores or create new dataframe
        if os.path.exists(cls.SCORES_FILE):
            scores_df = pd.read_csv(cls.SCORES_FILE)
            scores_df = pd.concat([scores_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            scores_df = pd.DataFrame([new_row])

        # Save updated scores
        scores_df.to_csv(cls.SCORES_FILE, index=False)

    @classmethod
    def load_all_scores(cls) -> Optional[pd.DataFrame]:
        """Load all test scores"""
        if os.path.exists(cls.SCORES_FILE):
            return pd.read_csv(cls.SCORES_FILE)
        return None

    @classmethod
    def load_test_details(cls, test_id: str) -> Optional[Dict]:
        """Load detailed results for a specific test"""
        results_file = os.path.join(cls.RESPONSES_DIR, f"{test_id}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return None

    @classmethod
    def get_model_statistics(cls) -> Optional[pd.DataFrame]:
        """Get statistics by model"""
        scores_df = cls.load_all_scores()
        if scores_df is not None and not scores_df.empty:
            stats = scores_df.groupby('model').agg({
                'score_percentage': ['mean', 'std', 'count'],
                'avg_response_time': 'mean'
            }).round(2)
            stats.columns = ['avg_score', 'std_score', 'test_count', 'avg_response_time']
            return stats.reset_index()
        return None