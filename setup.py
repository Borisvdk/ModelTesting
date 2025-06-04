from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-exam-tester",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for testing LLMs on multiple-choice exams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-exam-tester",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.32.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "llm-exam-tester=src.ui.app:main",
        ],
    },
)