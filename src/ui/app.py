import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models import OllamaModel
from src.core import DataLoader, TestRunner, ResultsManager

# Page config
st.set_page_config(
    page_title="LLM Exam Tester",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'questions_df' not in st.session_state:
    st.session_state.questions_df = None
if 'answers_df' not in st.session_state:
    st.session_state.answers_df = None
if 'exam_source' not in st.session_state:
    st.session_state.exam_source = 'default'


def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🎓 LLM Multiple-Choice Exam Tester</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: gray;'>Benchmark your Local LLMs with comprehensive testing</p>",
            unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🚀 Test Runner", "🏆 Leaderboard", "📊 Question Analytics", "📈 History", "ℹ️ About"])

    # Sidebar configuration
    setup_sidebar()

    # Tab 1: Test Runner
    with tab1:
        run_test_tab()

    # Tab 2: Leaderboard
    with tab2:
        show_leaderboard()

    # Tab 3: Question Analytics
    with tab3:
        show_question_analytics()

    # Tab 4: History
    with tab4:
        show_history()

    # Tab 5: About
    with tab5:
        show_about()


def setup_sidebar():
    """Setup sidebar with model selection and exam options"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = OllamaModel.list_available_models()

        if not available_models:
            st.error("No Ollama models found!")
            st.info("Make sure Ollama is running:")
            st.code("ollama serve", language="bash")
            st.info("Then download a model:")
            st.code("ollama pull llama3.2:3b", language="bash")
            st.session_state.selected_model = None
        else:
            st.session_state.selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=0,
                help="Choose the LLM model to test"
            )

            if st.button("🔄 Refresh Models", use_container_width=True):
                st.rerun()

        st.divider()

        # Exam selection
        st.subheader("📚 Exam Selection")
        exam_source = st.radio(
            "Choose exam source:",
            ["CCZT Exam", "Upload Custom"],
            index=0 if st.session_state.exam_source == 'default' else 1
        )

        if exam_source == "CCZT Exam":
            st.session_state.exam_source = 'default'
            # Load default exam automatically
            load_default_exam()
        else:
            st.session_state.exam_source = 'custom'
            st.info("Upload your exam files in the Test Runner tab")

        # Display current exam info
        if st.session_state.questions_df is not None:
            st.success(f"✅ Exam loaded: {len(st.session_state.questions_df)} questions")

        st.divider()

        # Quick stats
        st.subheader("📊 Quick Stats")
        scores_df = ResultsManager.load_all_scores()
        if scores_df is not None and not scores_df.empty:
            st.metric("Total Tests Run", len(scores_df))
            st.metric("Models Tested", scores_df['model'].nunique())
            st.metric("Avg Score", f"{scores_df['score_percentage'].mean():.1f}%")


def load_default_exam():
    """Load the default exam"""
    questions_df, answers_df = DataLoader.load_default_exam()
    if questions_df is not None and answers_df is not None:
        st.session_state.questions_df = questions_df
        st.session_state.answers_df = answers_df
        return True
    return False


def run_test_tab():
    """Test runner tab"""
    st.header("🚀 Test Runner")

    if st.session_state.exam_source == 'custom':
        # Custom exam upload section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📁 Upload Questions")
            questions_file = st.file_uploader(
                "Choose questions CSV",
                type=['csv'],
                help="CSV with columns: id, question, option_a, option_b, option_c, option_d"
            )

            if questions_file:
                try:
                    questions_df = pd.read_csv(questions_file)
                    st.session_state.questions_df = questions_df
                    st.success(f"✅ Loaded {len(questions_df)} questions")

                    with st.expander("Preview Questions"):
                        st.dataframe(questions_df.head())
                except Exception as e:
                    st.error(f"Error loading questions: {str(e)}")

        with col2:
            st.subheader("📁 Upload Answers")
            answers_file = st.file_uploader(
                "Choose answers CSV",
                type=['csv'],
                help="CSV with columns: id, correct_answer"
            )

            if answers_file:
                try:
                    answers_df = pd.read_csv(answers_file)
                    st.session_state.answers_df = answers_df
                    st.success(f"✅ Loaded {len(answers_df)} answers")

                    with st.expander("Preview Answers"):
                        st.dataframe(answers_df.head())
                except Exception as e:
                    st.error(f"Error loading answers: {str(e)}")

    # Test execution section
    st.divider()

    if st.session_state.questions_df is not None and st.session_state.answers_df is not None:
        # Show exam info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📝 **{len(st.session_state.questions_df)}** Questions Ready")
        with col2:
            st.info(f"🤖 **Model:** {st.session_state.selected_model or 'Not selected'}")
        with col3:
            st.info(f"📚 **Source:** {st.session_state.exam_source.title()}")

        # Run test button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ RUN TEST", type="primary", use_container_width=True,
                         disabled=not st.session_state.selected_model):
                if st.session_state.selected_model:
                    run_test(st.session_state.selected_model)
                else:
                    st.error("Please select a model first!")
    else:
        st.warning("⚠️ Please load an exam first (use sidebar to select default exam or upload custom files)")

    # Current test results
    if st.session_state.test_results:
        st.divider()
        display_current_results(st.session_state.test_results)


def run_test(model_name):
    """Run the test with enhanced progress tracking"""
    # Create placeholder for live updates
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()

        # Start timer
        start_time = time.time()

        try:
            runner = TestRunner()

            def progress_callback(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
                elapsed = time.time() - start_time
                time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s")

            # Run test
            results, score = runner.run_test(
                model_name,
                st.session_state.questions_df,
                st.session_state.answers_df,
                progress_callback
            )

            # Update UI
            progress_bar.progress(1.0)
            status_text.text("✅ Test completed!")
            total_time = time.time() - start_time
            time_text.text(f"⏱️ Total time: {total_time:.1f}s")

            # Store results
            st.session_state.test_results = {
                'model': model_name,
                'results': results,
                'score': score,
                'total_time': total_time
            }

            # Success message with balloons
            st.success(f"🎉 Test completed! Score: **{score:.1f}%**")
            if score >= 80:
                st.balloons()

        except Exception as e:
            st.error(f"❌ Error running test: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            time_text.empty()


def display_current_results(test_data):
    """Display current test results with enhanced visuals"""
    st.subheader("📊 Test Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", test_data['model'])

    with col2:
        # Color-coded score
        score = test_data['score']
        color = "#28a745" if score >= 80 else "#ffc107" if score >= 60 else "#dc3545"
        st.markdown(f"<div class='big-number' style='color: {color};'>{score:.1f}%</div>", unsafe_allow_html=True)
        st.caption("Score")

    with col3:
        correct = sum(r['is_correct'] for r in test_data['results'])
        total = len(test_data['results'])
        st.metric("Correct Answers", f"{correct}/{total}")

    with col4:
        avg_time = sum(r['response_time'] for r in test_data['results']) / len(test_data['results'])
        st.metric("Avg Response Time", f"{avg_time:.2f}s")

    # Detailed results
    with st.expander("📋 Detailed Question Results", expanded=True):
        results_df = pd.DataFrame(test_data['results'])

        # Format for display
        display_df = results_df[['question_id', 'question', 'extracted_answer',
                                 'correct_answer', 'is_correct', 'response_time']].copy()
        display_df['status'] = display_df['is_correct'].map({True: '✅', False: '❌'})
        display_df['response_time'] = display_df['response_time'].round(2).astype(str) + 's'

        # Rename columns
        display_df.columns = ['ID', 'Question', 'Model Answer', 'Correct Answer',
                              'Correct?', 'Time', 'Status']

        st.dataframe(
            display_df[['Status', 'ID', 'Question', 'Model Answer', 'Correct Answer', 'Time']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn(
                    "Question",
                    width="large",
                ),
            }
        )

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Detailed Results",
            csv,
            f"test_results_{test_data['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


def show_leaderboard():
    """Display model leaderboard"""
    st.header("🏆 Model Leaderboard")

    leaderboard = ResultsManager.get_leaderboard()

    if leaderboard is not None and not leaderboard.empty:
        # Add medals for top 3
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        leaderboard['medal'] = leaderboard['rank'].map(lambda x: medals.get(x, ""))

        # Create columns for metrics
        cols = st.columns(len(leaderboard) if len(leaderboard) < 5 else 5)

        for idx, (_, row) in enumerate(leaderboard.head(5).iterrows()):
            with cols[idx % 5]:
                medal = row['medal']
                st.markdown(f"### {medal} #{row['rank']} {row['model']}")
                st.metric("Score", f"{row['avg_score']:.1f}%")
                st.metric("Speed", f"{row['avg_response_time']:.2f}s")
                st.metric("Tests", row['test_count'])

        st.divider()

        # Full leaderboard table
        st.subheader("📊 Complete Rankings")

        display_df = leaderboard.copy()
        display_df['avg_score'] = display_df['avg_score'].round(1).astype(str) + '%'
        display_df['avg_response_time'] = display_df['avg_response_time'].round(2).astype(str) + 's'
        display_df['combined_score'] = display_df['combined_score'].round(1)

        st.dataframe(
            display_df[['medal', 'rank', 'model', 'avg_score', 'avg_response_time',
                        'combined_score', 'test_count']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "medal": st.column_config.TextColumn("", width="small"),
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "model": st.column_config.TextColumn("Model", width="medium"),
                "avg_score": st.column_config.TextColumn("Avg Score", width="small"),
                "avg_response_time": st.column_config.TextColumn("Avg Time", width="small"),
                "combined_score": st.column_config.NumberColumn("Combined Score", width="small"),
                "test_count": st.column_config.NumberColumn("Tests", width="small"),
            }
        )

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Score comparison
            fig = px.bar(
                leaderboard,
                x='model',
                y='avg_score',
                title='Average Scores by Model',
                labels={'avg_score': 'Average Score (%)', 'model': 'Model'},
                color='avg_score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Speed comparison
            fig = px.bar(
                leaderboard,
                x='model',
                y='avg_response_time',
                title='Average Response Time by Model',
                labels={'avg_response_time': 'Avg Response Time (s)', 'model': 'Model'},
                color='avg_response_time',
                color_continuous_scale='reds_r'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("🏃 No test results yet. Run some tests to see the leaderboard!")


def show_question_analytics():
    """Display question performance analytics"""
    st.header("📊 Question Performance Analytics")

    question_stats = ResultsManager.get_question_analytics()

    if question_stats is not None and not question_stats.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_success = question_stats['success_rate'].mean()
            st.metric("Avg Success Rate", f"{avg_success:.1f}%")

        with col2:
            hard_questions = len(question_stats[question_stats['difficulty'] == 'Hard'])
            st.metric("Hard Questions", hard_questions)

        with col3:
            easy_questions = len(question_stats[question_stats['difficulty'] == 'Easy'])
            st.metric("Easy Questions", easy_questions)

        with col4:
            total_attempts = question_stats['attempts'].sum()
            st.metric("Total Attempts", total_attempts)

        st.divider()

        # Difficulty distribution
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart of difficulty distribution
            difficulty_counts = question_stats['difficulty'].value_counts()
            fig = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title='Question Difficulty Distribution',
                color_discrete_map={'Easy': '#28a745', 'Medium': '#ffc107', 'Hard': '#dc3545'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Success rate distribution
            fig = px.histogram(
                question_stats,
                x='success_rate',
                nbins=20,
                title='Success Rate Distribution',
                labels={'success_rate': 'Success Rate (%)', 'count': 'Number of Questions'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed question table
        st.subheader("📋 Question Details")

        # Add color coding for difficulty
        def difficulty_color(val):
            colors = {'Easy': '#d4edda', 'Medium': '#fff3cd', 'Hard': '#f8d7da'}
            return f'background-color: {colors.get(val, "#ffffff")}'

        styled_df = question_stats.style.applymap(
            difficulty_color,
            subset=['difficulty']
        )

        st.dataframe(
            question_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "question_id": st.column_config.TextColumn("ID", width="small"),
                "question": st.column_config.TextColumn("Question", width="large"),
                "success_rate": st.column_config.NumberColumn("Success Rate (%)", width="small"),
                "difficulty": st.column_config.TextColumn("Difficulty", width="small"),
                "attempts": st.column_config.NumberColumn("Attempts", width="small"),
                "avg_response_time": st.column_config.NumberColumn("Avg Time (s)", width="small"),
                "models_tested": st.column_config.NumberColumn("Models Tested", width="small"),
                "common_mistake": st.column_config.TextColumn("Common Wrong Answer", width="medium"),
            }
        )

        # Hardest questions spotlight
        st.subheader("🎯 Hardest Questions")
        hardest = question_stats.nsmallest(5, 'success_rate')

        for _, q in hardest.iterrows():
            with st.expander(f"❌ {q['question']} (Success: {q['success_rate']}%)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Success Rate", f"{q['success_rate']}%")
                with col2:
                    st.metric("Attempts", q['attempts'])
                with col3:
                    st.metric("Common Mistake", q['common_mistake'])

    else:
        st.info("📊 No analytics available yet. Run some tests to see question performance!")


def show_history():
    """Display test history"""
    st.header("📈 Test History")

    scores_df = ResultsManager.load_all_scores()

    if scores_df is not None and not scores_df.empty:
        # Convert timestamp
        scores_df['timestamp'] = pd.to_datetime(scores_df['timestamp'])

        # Performance over time
        fig = px.line(
            scores_df.sort_values('timestamp'),
            x='timestamp',
            y='score_percentage',
            color='model',
            title='Model Performance Over Time',
            labels={'score_percentage': 'Score (%)', 'timestamp': 'Date'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Recent tests table
        st.subheader("📋 Recent Tests")

        recent_tests = scores_df.sort_values('timestamp', ascending=False).head(20).copy()
        recent_tests['timestamp'] = recent_tests['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        recent_tests['score_percentage'] = recent_tests['score_percentage'].round(1).astype(str) + '%'
        recent_tests['avg_response_time'] = recent_tests['avg_response_time'].round(2).astype(str) + 's'

        st.dataframe(
            recent_tests[['test_id', 'model', 'timestamp', 'score_percentage', 'avg_response_time']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "test_id": st.column_config.TextColumn("Test ID", width="medium"),
                "model": st.column_config.TextColumn("Model", width="medium"),
                "timestamp": st.column_config.TextColumn("Date/Time", width="medium"),
                "score_percentage": st.column_config.TextColumn("Score", width="small"),
                "avg_response_time": st.column_config.TextColumn("Avg Time", width="small"),
            }
        )

        # Download all results
        csv = scores_df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results",
            csv,
            f"all_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

    else:
        st.info("📈 No test history yet. Start running tests to build your history!")


def show_about():
    """Display about information"""
    st.header("ℹ️ About LLM Exam Tester")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Purpose

        This framework helps you evaluate and compare the performance of different Local LLMs 
        on multiple-choice exam questions. It's perfect for:

        - **Benchmarking** different model sizes and types
        - **Testing** model knowledge across various domains
        - **Comparing** speed vs accuracy trade-offs
        - **Tracking** model improvements over time

        ### 🔧 How It Works

        1. **Load an Exam**: Use the default general knowledge exam or upload your own
        2. **Select a Model**: Choose from any Ollama-compatible model
        3. **Run the Test**: The framework will query each question and record responses
        4. **Analyze Results**: View scores, response times, and detailed analytics

        ### 📊 Metrics Explained

        - **Score**: Percentage of questions answered correctly
        - **Response Time**: Average time taken per question
        - **Combined Score**: Weighted score (70% accuracy, 30% speed)
        - **Difficulty Rating**: Based on success rate across all tests

        ### 🚀 Tips for Best Results

        - Test multiple models to find the best balance of speed and accuracy
        - Run multiple tests to account for variability
        - Use consistent hardware for fair comparisons
        - Consider model size vs performance trade-offs
        """)

    with col2:
        st.markdown("""
        ### 📋 Quick Stats
        """)

        # Show some aggregate stats
        scores_df = ResultsManager.load_all_scores()
        if scores_df is not None and not scores_df.empty:
            st.metric("Total Tests Run", len(scores_df))
            st.metric("Unique Models", scores_df['model'].nunique())
            st.metric("Questions in Default Exam", "20")

            best_model = scores_df.groupby('model')['score_percentage'].mean().idxmax()
            best_score = scores_df.groupby('model')['score_percentage'].mean().max()
            st.success(f"🏆 Best Model: {best_model} ({best_score:.1f}%)")

        st.markdown("""
        ### 🔗 Links

        - [Ollama](https://ollama.ai/)
        - [Streamlit](https://streamlit.io/)
        - [GitHub Repository](#)

        ### 🤝 Contributing

        Found a bug or have a feature request? 
        Please open an issue on GitHub!
        """)


if __name__ == "__main__":
    main()