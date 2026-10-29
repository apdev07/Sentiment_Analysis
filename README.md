# 🧠 Sentiment Analysis System

A comprehensive Natural Language Processing (NLP) web application that classifies review sentiments using Recurrent Neural Networks (LSTM) trained on the IMDB dataset.

---

## 📌 Project Overview

- **Interactive Web Interface**: Built with modern Streamlit styling, intuitive tabs, and real-time response.
- **Deep Learning Model**: An LSTM (Long Short-Term Memory) neural network with word embeddings to capture context and long-range semantic dependencies.
- **Single & Batch Processing**: Real-time single review classification with confidence meters, plus bulk CSV upload and processing.
- **Exportable Analytics**: Downloadable reports in Markdown and CSV formats for audit trails.

---

## 📂 Project Structure

```text
Sentiment-Analysis/
├── sentiment_app.py                # Main Streamlit web application
├── app.py                          # Application entrypoint launcher
├── Sentiment_Model_Training.ipynb  # Jupyter Notebook for data prep, training & evaluation
├── model.h5                        # Saved trained LSTM model weights
├── tokenizer.pkl                   # Fitted Keras text tokenizer
├── IMDB Dataset.csv                # 50k labeled movie reviews dataset
├── text_sources.py                 # CSV ingestion and sanitization utilities
├── requirements.txt                # Python dependencies
├── README.md                       # Project manual and setup guide
└── tests/
    └── test_text_sources.py        # Automated test suite for data ingestion
```

---

## ⚙️ How to Run the Project

Follow these simple steps to run the project on your local machine:

### 1. Prerequisites
- Python 3.9 to 3.13 installed
- Command Prompt, PowerShell, or Terminal

### 2. Open Project Folder
Open your terminal or PowerShell and navigate to the project directory:
```bash
cd "c:\Users\sanja\Downloads\Sentiment-Analysis"
```

### 3. (Recommended) Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Web Application
Run the Streamlit server:
```bash
streamlit run sentiment_app.py
```
*(You can also run `streamlit run app.py`)*

Once started, Streamlit will automatically open your default browser at:
```text
http://localhost:8501
```

---

## 📓 Model Training & Experimentation


1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Sentiment_Model_Training.ipynb`.
3. The notebook documents the complete workflow:
   - Data loading and exploratory data analysis (EDA).
   - Text cleaning, tokenization, and padding.
   - Building the LSTM architecture with TensorFlow/Keras.
   - Training curves (loss/accuracy) and model evaluation.
   - Exporting `model.h5` and `tokenizer.pkl`.

---

## 🧪 Running Automated Tests

To verify that the CSV ingestion and text parsing utilities are working correctly:
```bash
python -m unittest discover -s tests
```
