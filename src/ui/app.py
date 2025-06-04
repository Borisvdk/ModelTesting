import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models import OllamaModel
from src.core import DataLoader, TestRunner, ResultsManager

# Page config
st.set_page_config(
    page_title="LLM Exam Tester",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'questions_df' not in st.session_state:
    st.session_state.questions_df = None
if 'answers_df' not in st.session_state:
    st.session_state.answers_df = None


def main():
    st.title("🎓 LLM Multiple-Choice Exam Tester")
    st.markdown("Evaluate Local LLMs on multiple-choice exam questions")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        st.subheader("Model Selection")
        available_models = OllamaModel.list_available_models()

        if not available_models:
            st.error("No Ollama models found. Make sure Ollama is running.")
            st.code("ollama pull llama3.2:3b", language="bash")
        else:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=0
            )

            if st.button("🔄 Refresh Models"):
                st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Data Upload")

        # Questions upload
        questions_file = st.file_uploader(
            "Upload Questions CSV",
            type=['csv'],
            help="CSV with columns: id, question, option_a, option_b, option_c, option_d"
        )

        if questions_file:
            try:
                questions_df = pd.read_csv(questions_file)
                st.session_state.questions_df = questions_df
                st.success(f"✅ Loaded {len(questions_df)} questions")

                # Preview
                with st.expander("Preview Questions"):
                    st.dataframe(questions_df.head())
            except Exception as e:
                st.error(f"Error loading questions: {str(e)}")

        # Answers upload
        answers_file = st.file_uploader(
            "Upload Answers CSV",
            type=['csv'],
            help="CSV with columns: id, correct_answer"
        )

        if answers_file:
            try:
                answers_df = pd.read_csv(answers_file)
                st.session_state.answers_df = answers_df
                st.success(f"✅ Loaded {len(answers_df)} answers")

                # Preview
                with st.expander("Preview Answers"):
                    st.dataframe(answers_df.head())
            except Exception as e:
                st.error(f"Error loading answers: {str(e)}")

    with col2:
        st.header("🚀 Test Execution")

        # Check if data is loaded
        if st.session_state.questions_df is not None and st.session_state.answers_df is not None:
            # Validate data
            loader = DataLoader()
            if loader.validate_data(st.session_state.questions_df, st.session_state.answers_df):
                st.info(f"Ready to test with {len(st.session_state.questions_df)} questions")

                # Run test button
                if st.button("▶️ RUN TEST", type="primary", use_container_width=True):
                    if not available_models:
                        st.error("No models available. Please install Ollama and download a model.")
                    else:
                        run_test(selected_model)
            else:
                st.error("Data validation failed. Please check your files.")
        else:
            st.warning("Please upload both questions and answers files")

    # Results section
    st.header("📊 Results")

    # Current test results
    if st.session_state.test_results:
        display_current_results(st.session_state.test_results)

    # Historical results
    display_historical_results()


def run_test(model_name):
    """Run the test with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        runner = TestRunner()

        def progress_callback(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)

        # Run test
        results, score = runner.run_test(
            model_name,
            st.session_state.questions_df,
            st.session_state.answers_df,
            progress_callback
        )

        # Update UI
        progress_bar.progress(1.0)
        status_text.text("Test completed!")

        # Store results
        st.session_state.test_results = {
            'model': model_name,
            'results': results,
            'score': score
        }

        st.success(f"✅ Test completed! Score: {score:.1f}%")
        st.balloons()

    except Exception as e:
        st.error(f"Error running test: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_current_results(test_data):
    """Display current test results"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", test_data['model'])

    with col2:
        st.metric("Score", f"{test_data['score']:.1f}%")

    with col3:
        correct = sum(r['is_correct'] for r in test_data['results'])
        total = len(test_data['results'])
        st.metric("Correct Answers", f"{correct}/{total}")

    # Detailed results
    with st.expander("Detailed Results"):
        results_df = pd.DataFrame(test_data['results'])

        # Format for display
        display_df = results_df[['question_id', 'question', 'extracted_answer',
                                 'correct_answer', 'is_correct', 'response_time']]
        display_df['status'] = display_df['is_correct'].map({True: '✅', False: '❌'})
        display_df['response_time'] = display_df['response_time'].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


def display_historical_results():
    """Display historical test results and comparisons"""
    scores_df = ResultsManager.load_all_scores()

    if scores_df is not None and not scores_df.empty:
        st.subheader("Historical Performance")

        # Model comparison chart
        model_stats = ResultsManager.get_model_statistics()
        if model_stats is not None:
            fig = px.bar(
                model_stats,
                x='model',
                y='avg_score',
                error_y='std_score',
                title='Average Score by Model',
                labels={'avg_score': 'Average Score (%)', 'model': 'Model'},
                color='avg_score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Recent tests table
        st.subheader("Recent Tests")
        recent_tests = scores_df.sort_values('timestamp', ascending=False).head(10)

        # Format for display
        display_cols = ['test_id', 'model', 'timestamp', 'score_percentage', 'avg_response_time']
        recent_tests['timestamp'] = pd.to_datetime(recent_tests['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent_tests['score_percentage'] = recent_tests['score_percentage'].round(1)
        recent_tests['avg_response_time'] = recent_tests['avg_response_time'].round(2)

        st.dataframe(
            recent_tests[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # Performance over time
        if len(scores_df) > 1:
            scores_df['timestamp'] = pd.to_datetime(scores_df['timestamp'])

            fig = px.line(
                scores_df.sort_values('timestamp'),
                x='timestamp',
                y='score_percentage',
                color='model',
                title='Performance Over Time',
                labels={'score_percentage': 'Score (%)', 'timestamp': 'Date'},
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical results available yet. Run your first test!")


if __name__ == "__main__":
    main()