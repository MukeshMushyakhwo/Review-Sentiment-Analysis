import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained logistic regression model
lr_model = joblib.load("Model/sentiment_analysis_model.pkl")

# Load the TfidfVectorizer
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")

def classify_sentiment(review, vectorizer):
    # Preprocess the input review text
    review = [review.lower()]
    review = vectorizer.transform(review)

    # Predict the sentiment using the model
    probability = lr_model.predict_proba(review)[0]
    positive_probability = probability[1]
    negative_probability = probability[0]

    # Determine the sentiment based on probabilities
    if positive_probability > negative_probability:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment

# Main Streamlit app function
def main():
    st.title("Review Sentiment Classification App")

    # Text input for user to enter review
    review_input = st.text_area("Enter your review here:")

    if st.button("Classify Sentiment"):
        if review_input.strip():  # Check if the input is not empty or contains only whitespace
            # Classify the sentiment and display the result
            sentiment = classify_sentiment(review_input, vectorizer)
            st.write(f"The sentiment of the review is: {sentiment.capitalize()}")

            # Display sentiment score and emoji
            if sentiment == "positive":
                emoji_html = '<span style="font-size: 36px;">👍</span>'
                st.markdown(emoji_html, unsafe_allow_html=True)
            else:
                emoji_html = '<span style="font-size: 36px;">👎</span>'
                st.markdown(emoji_html, unsafe_allow_html=True)
        else:
            st.write("Please enter a review to classify.")

if __name__ == "__main__":
    main()
