import streamlit as st
import joblib

model = joblib.load("./models/language_detector_model.pkl")
tfidf = joblib.load("./models/tfidf_vectorizer.pkl")
le = joblib.load("./models/label_encoder.pkl")


def detect_language(text):
    text_vectorized = tfidf.transform([text])
    prediction = model.predict(text_vectorized)
    language = le.inverse_transform(prediction)[0]
    return language


st.set_page_config(page_title="Language Detector", page_icon="🌍")
st.title("🌍 Language Detection using Machine Learning")

st.markdown("""
This app detects the **language** of your input sentence using a trained **Multinomial Naive Bayes** model.
Just type in any sentence, and the model will tell you what language it is.
""")

input_text = st.text_area("Enter your text here:", height=100)

if st.button("Detect Language"):
    if input_text.strip() == "":
        st.error("Please enter some text to detect the language.")
    else:
        with st.spinner("Detecting..."):
            language = detect_language(input_text)
            st.success(f"The detected language is: **{language}**")