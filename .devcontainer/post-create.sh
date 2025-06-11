#!/bin/bash
set -e

echo "🚀 Setting up LLM Exam Tester development environment..."

# Install project dependencies
echo "📦 Installing Python dependencies..."
pip install --user -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/exams data/results/responses config

# Check if sample data exists, if not create it
if [ ! -f "data/exams/questions.csv" ]; then
    echo "📝 Creating sample exam data..."
    cat > data/exams/questions.csv << 'EOF'
id,question,option_a,option_b,option_c,option_d
Q001,"What is the capital of France?","London","Paris","Berlin","Madrid"
Q002,"Which element has symbol 'Au'?","Silver","Copper","Gold","Aluminum"
Q003,"What is 2+2?","3","4","5","6"
Q004,"Who painted the Mona Lisa?","Van Gogh","Da Vinci","Picasso","Rembrandt"
Q005,"What is the largest planet in our solar system?","Earth","Mars","Jupiter","Saturn"
Q006,"Which year did World War II end?","1943","1944","1945","1946"
Q007,"What is the chemical formula for water?","CO2","H2O","O2","N2"
Q008,"Who wrote 'Romeo and Juliet'?","Charles Dickens","William Shakespeare","Jane Austen","Mark Twain"
Q009,"What is the speed of light in vacuum?","300,000 km/s","299,792 km/s","250,000 km/s","350,000 km/s"
Q010,"Which country has the largest population?","India","United States","China","Russia"
Q011,"What is the smallest unit of matter?","Molecule","Atom","Electron","Quark"
Q012,"How many continents are there?","5","6","7","8"
Q013,"What is the capital of Japan?","Seoul","Beijing","Tokyo","Bangkok"
Q014,"Which gas do plants absorb from the atmosphere?","Oxygen","Nitrogen","Carbon Dioxide","Hydrogen"
Q015,"What is the largest ocean on Earth?","Atlantic","Indian","Arctic","Pacific"
Q016,"Who developed the theory of relativity?","Newton","Einstein","Galileo","Hawking"
Q017,"What is the main ingredient in glass?","Carbon","Silicon","Sand","Limestone"
Q018,"How many bones are in the adult human body?","106","206","306","406"
Q019,"What is the currency of the United Kingdom?","Euro","Dollar","Pound","Franc"
Q020,"Which planet is known as the Red Planet?","Venus","Mars","Jupiter","Mercury"
EOF

    cat > data/exams/answers.csv << 'EOF'
id,correct_answer
Q001,B
Q002,C
Q003,B
Q004,B
Q005,C
Q006,C
Q007,B
Q008,B
Q009,B
Q010,C
Q011,D
Q012,C
Q013,C
Q014,C
Q015,D
Q016,B
Q017,C
Q018,B
Q019,C
Q020,B
EOF
fi

# Create config file if it doesn't exist
if [ ! -f "config/prompts.yaml" ]; then
    echo "⚙️ Creating config file..."
    cat > config/prompts.yaml << 'EOF'
system_prompt: |
  You are taking a multiple-choice exam. You will be given a question with four options labeled A, B, C, and D.
  
  Your task is to select the correct answer and respond with ONLY the letter (A, B, C, or D).
  Do not include any explanation, punctuation, or additional text.
  
  Example:
  Question: What is 2+2?
  A) 3
  B) 4
  C) 5
  D) 6
  
  Correct response: B

question_template: |
  Question: {question}
  A) {option_a}
  B) {option_b}
  C) {option_c}
  D) {option_d}
EOF
fi

# Start Ollama service in the background
echo "🦙 Starting Ollama service..."
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Pull a default model (optional - commented out to save time/bandwidth)
# echo "📥 Pulling default model (this may take a while)..."
# ollama pull llama3.2:3b || echo "⚠️ Could not pull model. Please run 'ollama pull llama3.2:3b' manually."

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 Quick Start Guide:"
echo "1. The Ollama service is running in the background"
echo "2. Pull a model: ollama pull llama3.2:3b"
echo "3. Run the app: streamlit run src/ui/app.py"
echo "4. Open http://localhost:8501 in your browser"
echo ""
echo "Happy testing! 🚀"