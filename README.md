<div align="center">

# 🎓 LLM Multiple-Choice Exam Tester

### *Benchmark Your Local LLMs with Style*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/ollama-compatible-black?style=for-the-badge)](https://ollama.ai/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

**Test** • **Compare** • **Analyze** • **Improve**

[Features](#-features) • [Quick Start](#-quick-start) • [Demo](#-demo) • [Documentation](#-documentation)

---

</div>

## 🌟 Why This Framework?

Ever wondered how different LLMs perform on standardized tests? This framework makes it **dead simple** to:

- 🚀 **Test any Ollama model** on multiple-choice questions
- 📊 **Compare performance** across different models
- 🏆 **Track improvements** with a built-in leaderboard
- 📈 **Analyze question difficulty** to understand model weaknesses
- 💾 **Save all results** for future analysis

### 🐳 Why Docker?

We use Docker to eliminate setup headaches:
- **Zero Python configuration** - No version conflicts
- **Ollama pre-installed** - Works immediately  
- **Consistent environment** - Same setup for everyone
- **One command to start** - `docker-compose up -d`
- **Cross-platform** - Works on Windows, Mac, Linux

<div align="center">
  <img src="https://via.placeholder.com/800x400/1a1a2e/eee?text=LLM+Exam+Tester+Screenshot" alt="Screenshot" width="800">
  <p><i>Clean, intuitive interface powered by Streamlit</i></p>
</div>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Features
- **Auto-loaded Sample Exams** - Start testing immediately
- **Custom Exam Upload** - Test on your own questions
- **Real-time Progress** - Watch as models think
- **Detailed Analytics** - Deep dive into results
- **Model Leaderboard** - See who's winning
- **Dev Container Support** - One-click setup

</td>
<td width="50%">

### 📊 Advanced Analytics
- **Question Performance Matrix** - Which questions stump models?
- **Response Time Analysis** - Speed vs accuracy insights
- **Historical Tracking** - Performance over time
- **Export Capabilities** - Take your data anywhere
- **Beautiful Visualizations** - Powered by Plotly
- **Zero Configuration** - Works out of the box

</td>
</tr>
</table>

## 🚀 Quick Start

### 🐳 Docker Compose (Recommended - Zero Setup!)

The **easiest way** to get started - everything runs in Docker!

```powershell
# 1. Clone the repository
git clone https://github.com/Borisvdk/ModelTesting.git
cd llm-exam-tester

# 2. Start the application
docker-compose up -d

# 3. Wait ~30 seconds for setup, then open
# http://localhost:8501

# 4. Pull a model (run this in a new terminal)
docker-compose exec llm-exam-tester ollama pull llama3.2:3b