# 🌍 Language Detection using Machine Learning

This project is a simple and effective machine learning system that detects the **language** of a given text input using text classification techniques. It uses a **TF-IDF Vectorizer** and a **Multinomial Naive Bayes** classifier to identify languages based on text features.

---

## 🚀 Features

- Detects language from raw text input
- Trained on a labeled dataset with `Text` and `Language`
- TF-IDF vectorization for text representation
- Multinomial Naive Bayes for classification
- Custom input prediction
- Model and vectorizer saving for deployment

---

## 📁 Project Structure

    language-detection-ml/
    │
    ├── datasets/
    │ └── Language_Detection.csv # Training dataset
    │
    ├── models/
    │ ├── language_detector_model.pkl # Trained ML model
    │ ├── tfidf_vectorizer.pkl # TF-IDF Vectorizer
    │ └── label_encoder.pkl # Label encoder for inverse transform
    │
    ├── notebooks/
    │ └── preprocessing.ipynb # Jupyter notebook for training and testing
    │
    └── README.md # Project documentation
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

Author
Prakash L Waddar

AI & Data Science Enthusiast | Full Stack Developer

LinkedIn: https://www.linkedin.com/in/prakash-l-waddar/
GitHub: https://github.com/prakashwaddar628

Feel free to reach out if you're interested in collaborating on other projects or need assistance with implementation.