# 🌍 Language Detection using Machine Learning

This project is a simple and effective machine learning system that detects the **language** of a given text input using text classification techniques. It uses a **TF-IDF Vectorizer** and a **Multinomial Naive Bayes** classifier to identify languages based on text features.

---

## 🚀 Features

    -Detects text language among 11 languages
    -Trained on labeled text samples
    -TF-IDF vectorization + Naive Bayes model
    -Evaluation via Confusion Matrix
    -Real-time predictions
    -Streamlit Web UI for input and output
    -Model and vectorizer saving for reuse

## 🧠 Languages Supported
    1.Arabic
    2.Dutch
    3.English
    4.French
    5.German
    6.Italian
    7.Portuguese
    8.Russian
    9.Spanish
    10.Swedish
    11.Turkish

---

## 📁 Project Structure

    language-detector/
    │
    ├── data/
    │   └── Language_Detection.csv         # Dataset used for training
    │
    ├── models/
    │   ├── language_detector_model.pkl    # Trained Naive Bayes model
    │   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
    │   └── label_encoder.pkl              # Label encoder for output decoding
    │
    ├── app.py                             # Streamlit app for live prediction
    ├── language_detection.ipynb           # Jupyter notebook for training + evaluation
    ├── README.md                          # Project documentation (you are here)
    └── requirements.txt                   # Python dependencies

---

## 🧠 Tech Stack

- **Python 3.11+**
- **Pandas** – Data handling
- **Scikit-learn** – ML model, TF-IDF, train-test split
- **Seaborn / Matplotlib** – Confusion matrix visualization
- **Joblib** – Model persistence

---

## 🔄 How it Works

1. **Load dataset** with columns `Text` and `Language`.
2. **Preprocess and vectorize** using `TfidfVectorizer`.
3. **Encode labels** using `LabelEncoder`.
4. **Split dataset** into training and testing sets.
5. **Train model** with `Multinomial Naive Bayes (MultinomialNB) classifier`.
6. **Evaluate** using accuracy and a confusion matrix.
7. **Predict** language for any input text using a simple function.

---
## 🚀 How to Run

# 1. Clone the repository
    git clone https://github.com/yourusername/language-detector.git
    cd language-detector

# 2. Run Streamlit app
    streamlit run app.py


## ▶️ Usage

### 📦 Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn joblib

**Run in Jupyter Notebook**

Open preprocessing.ipynb and run all cells to:
- Train the model
- Evaluate performance
- Predict on custom input
- Save trained files

📈 Model Performance
The model was evaluated using a confusion matrix, showing good performance across all supported languages. TF-IDF helps capture language-specific vocabulary patterns, and MultinomialNB handles text classification efficiently.

💡 Future Improvements
Add support for more languages (e.g., Hindi, Japanese, Chinese)

Integrate FastText or XLM-Roberta for larger-scale detection

Build REST API with Flask or FastAPI

Deploy app online with Streamlit Cloud / Hugging Face Spaces

Author
Prakash L Waddar

AI & Data Science Enthusiast | Full Stack Developer

LinkedIn: https://www.linkedin.com/in/prakash-l-waddar/
GitHub: https://github.com/prakashwaddar628

✉️ Feel free to reach out if you're interested in collaborating on other projects or need assistance with implementation.