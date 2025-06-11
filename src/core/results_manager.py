import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging
from collections import defaultdict

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

    @classmethod
    def get_leaderboard(cls) -> Optional[pd.DataFrame]:
        """Get model leaderboard ranked by performance"""
        model_stats = cls.get_model_statistics()
        if model_stats is not None:
            # Calculate combined score (70% accuracy, 30% speed)
            # Normalize response time (inverse, so faster is better)
            max_time = model_stats['avg_response_time'].max()
            model_stats['speed_score'] = (max_time - model_stats['avg_response_time']) / max_time * 100

            # Combined score
            model_stats['combined_score'] = (
                    model_stats['avg_score'] * 0.7 +
                    model_stats['speed_score'] * 0.3
            ).round(2)

            # Add rank
            model_stats['rank'] = model_stats['combined_score'].rank(ascending=False, method='min').astype(int)

            # Sort by rank
            leaderboard = model_stats.sort_values('rank')

            return leaderboard[['rank', 'model', 'avg_score', 'avg_response_time',
                                'combined_score', 'test_count']]
        return None

    @classmethod
    def get_question_analytics(cls) -> Optional[pd.DataFrame]:
        """Analyze performance by question across all tests"""
        question_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'correct_count': 0,
            'total_response_time': 0,
            'models_attempted': set(),
            'question_text': '',
            'wrong_answers': defaultdict(int)
        })

        # Load all test results
        if not os.path.exists(cls.RESPONSES_DIR):
            return None

        for filename in os.listdir(cls.RESPONSES_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(cls.RESPONSES_DIR, filename)
                with open(filepath, 'r') as f:
                    test_data = json.load(f)

                for result in test_data['results']:
                    q_id = result['question_id']
                    stats = question_stats[q_id]

                    stats['total_attempts'] += 1
                    stats['correct_count'] += int(result['is_correct'])
                    stats['total_response_time'] += result['response_time']
                    stats['models_attempted'].add(test_data['model'])
                    stats['question_text'] = result['question']

                    if not result['is_correct'] and result['extracted_answer']:
                        stats['wrong_answers'][result['extracted_answer']] += 1

        # Convert to DataFrame
        if question_stats:
            data = []
            for q_id, stats in question_stats.items():
                success_rate = (stats['correct_count'] / stats['total_attempts'] * 100) if stats[
                                                                                               'total_attempts'] > 0 else 0
                avg_time = stats['total_response_time'] / stats['total_attempts'] if stats['total_attempts'] > 0 else 0

                # Determine difficulty
                if success_rate >= 80:
                    difficulty = "Easy"
                elif success_rate >= 50:
                    difficulty = "Medium"
                else:
                    difficulty = "Hard"

                # Most common wrong answer
                if stats['wrong_answers']:
                    most_common_wrong = max(stats['wrong_answers'].items(), key=lambda x: x[1])
                    common_mistake = f"{most_common_wrong[0]} ({most_common_wrong[1]} times)"
                else:
                    common_mistake = "N/A"

                data.append({
                    'question_id': q_id,
                    'question': stats['question_text'][:50] + '...' if len(stats['question_text']) > 50 else stats[
                        'question_text'],
                    'success_rate': round(success_rate, 1),
                    'difficulty': difficulty,
                    'attempts': stats['total_attempts'],
                    'avg_response_time': round(avg_time, 2),
                    'models_tested': len(stats['models_attempted']),
                    'common_mistake': common_mistake
                })

            df = pd.DataFrame(data)
            return df.sort_values('success_rate', ascending=True)

        return None

    @classmethod
    def get_recent_tests(cls, limit: int = 10) -> Optional[pd.DataFrame]:
        """Get most recent test results"""
        scores_df = cls.load_all_scores()
        if scores_df is not None and not scores_df.empty:
            scores_df['timestamp'] = pd.to_datetime(scores_df['timestamp'])
            return scores_df.sort_values('timestamp', ascending=False).head(limit)
        return None

    @classmethod
    def get_results_matrix(cls, mode: str = "latest") -> Optional[pd.DataFrame]:
        """
        Generate a matrix view of results for export.

        Args:
            mode: "latest" for most recent test per model, "all" for all tests

        Returns:
            DataFrame with models as rows and questions as columns
        """
        if not os.path.exists(cls.RESPONSES_DIR):
            return None

        # Collect all test results
        all_results = []

        for filename in os.listdir(cls.RESPONSES_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(cls.RESPONSES_DIR, filename)
                with open(filepath, 'r') as f:
                    test_data = json.load(f)

                    # Create a row for this test
                    row_data = {
                        'model': test_data['model'],
                        'test_id': test_data['test_id'],
                        'timestamp': test_data['timestamp'],
                        'total': test_data['summary']['correct_answers'],
                        'total_questions': test_data['summary']['total_questions']
                    }

                    # Add question results
                    question_results = {}
                    for result in test_data['results']:
                        # Extract question number from ID (e.g., Q001 -> 1)
                        q_num = int(result['question_id'].replace('Q', ''))
                        question_results[q_num] = 1 if result['is_correct'] else 0

                    row_data['question_results'] = question_results
                    all_results.append(row_data)

        if not all_results:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # If mode is "latest", keep only most recent test per model
        if mode == "latest":
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').groupby('model').last().reset_index()

        # Find max question number
        max_question = 0
        for _, row in df.iterrows():
            if row['question_results']:
                max_question = max(max_question, max(row['question_results'].keys()))

        # Create matrix format
        matrix_data = []
        for _, row in df.iterrows():
            matrix_row = {
                'Model': row['model'],
                'Type': 'Lokaal',  # Local
                'RAG': 'No',  # RAG support (future feature)
                '1by1': 'Yes',  # Questions asked one by one
                'Total': row['total']
            }

            # Add question columns
            for i in range(1, max_question + 1):
                matrix_row[str(i)] = row['question_results'].get(i, 0)

            matrix_data.append(matrix_row)

        matrix_df = pd.DataFrame(matrix_data)

        # Sort by total score (descending)
        matrix_df = matrix_df.sort_values('Total', ascending=False)

        return matrix_df

    @classmethod
    def export_results_matrix(cls, filename: str, mode: str = "latest") -> bool:
        """Export results in matrix format to CSV"""
        matrix_df = cls.get_results_matrix(mode)
        if matrix_df is not None:
            matrix_df.to_csv(filename, index=False)
            return True
        return False