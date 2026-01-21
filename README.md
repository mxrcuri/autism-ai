# Autism-AI

A machine learning pipeline for early autism screening using behavioral signals inspired by the DREAM dataset.  
The system extracts pose, gaze, and motion features and trains a Temporal Convolutional Network (TCN) autoencoder to model typical behavior and detect deviations.

---

## 🚀 Setup & Run Instructions

### 1️⃣ Clone the repository

git clone https://github.com/mxrcuri/autism-ai.git
cd autism-ai

### 2️⃣ Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

### ▶️ Running the Pipeline

All scripts are executed from the backend directory:
python -m scripts.run_step2
python -m scripts.run_step3
python -m scripts.run_step4
python -m scripts.run_step5
