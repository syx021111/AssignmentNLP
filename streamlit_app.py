import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Ensure the VADER lexicon and stopwords are downloaded
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load stop words and initialize the stemmer
stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')

def remove_stopwords(text):
    """ Remove stop words from a given text """
    return ' '.join([word for word in text.split() if word.lower() not in stop])

def apply_stemming(text):
    """ Apply stemming to the given text """
    return ' '.join([sno.stem(word) for word in text.split()])

def preprocess_text(text):
    """ Convert text to lowercase, remove stopwords, and apply stemming """
    text = text.lower()  # Convert to lowercase
    text = remove_stopwords(text)
    text = apply_stemming(text)
    return text

def create_pie_chart(textblob_results, vader_results):
    """ Create and display pie charts comparing TextBlob and VADER results """
    # Count the occurrences of each sentiment for TextBlob and VADER
    textblob_counts = pd.Series(textblob_results).value_counts()
    vader_counts = pd.Series(vader_results).value_counts()

    # Plot pie chart for TextBlob results
    st.write("### TextBlob Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.2f%%', startangle=140, colors=['#DC381F', '#01F9C6', '#2B65EC'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('TextBlob Sentiment Distribution')
    st.pyplot(fig1)

    # Plot pie chart for VADER results
    st.write("### NLTK VADER Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    ax2.pie(vader_counts, labels=vader_counts.index, autopct='%1.2f%%', startangle=140, colors=['#DC381F', '#01F9C6', '#2B65EC'])
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('VADER Sentiment Distribution')
    st.pyplot(fig2)

def create_wordcloud(df, sentiment_column):
    """ Generate word clouds for each sentiment category """
    # For each unique sentiment (Positive, Negative, Neutral)
    for sentiment in df[sentiment_column].unique():
        subset = df[df[sentiment_column] == sentiment]
        
        # Ensure the column used for word cloud creation contains the text (cleaned_text in this case)
        if 'cleaned_text' in df.columns:
            text = " ".join(subset['cleaned_text'].tolist())  # Combine all text entries for this sentiment
        else:
            st.error("No valid text column found for word cloud generation.")
            return

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud using Streamlit
        st.write(f"### Most Common Words in {sentiment.capitalize()} ({sentiment_column})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.title(f'Most Common Words in {sentiment.capitalize()} ({sentiment_column})')
        st.pyplot(fig)


def apply_tfidf_transformer(text_list):
    """ Apply TF-IDF vectorizer to the text """
    vectorizer = TfidfVectorizer()
    transformed_text = vectorizer.fit_transform(text_list).toarray()  # Transform text to TF-IDF
    return transformed_text, vectorizer

def main():
    # Title of your web app
    st.title("Sentiment Analysis with Model Evaluation, Pie Charts, and Word Clouds")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        user_input = st.text_area("Enter text to analyze sentiment:")

        if st.button('Analyze Sentiment'):
            if user_input.strip():  # Check if input is not empty
                processed_input = preprocess_text(user_input)

                textblob_result = analyze_sentiment_textblob(processed_input)
                vader_result = analyze_sentiment_vader(processed_input)
                st.write(f"TextBlob Sentiment: {textblob_result}")
                st.write(f"NLTK VADER Sentiment: {vader_result}")

                # Create pie chart for single input
                create_pie_chart([textblob_result], [vader_result])

                # Apply TF-IDF transformation
                tfidf_result, tfidf_vectorizer = apply_tfidf_transformer([processed_input])
                st.write("TF-IDF Vectorized Result:")
                st.write(tfidf_result)
            else:
                st.error("Please enter some text for analysis.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            # Read CSV file
            data = pd.read_csv(uploaded_file)

            if 'text' in data.columns and 'sentiment' in data.columns:
                # Preprocess the text column
                data['cleaned_text'] = data['text'].apply(preprocess_text)
                
                # Apply TF-IDF transformation to the cleaned text
                tfidf_result, tfidf_vectorizer = apply_tfidf_transformer(data['cleaned_text'].tolist())

                # Label encoding for the target sentiment column
                le = LabelEncoder()
                data['encoded_sentiment'] = le.fit_transform(data['sentiment'])

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(tfidf_result, data['encoded_sentiment'], test_size=0.2, random_state=0)

                # Sentiment analysis using TextBlob and VADER for comparison
                data['TextBlob Sentiment'] = data['cleaned_text'].apply(analyze_sentiment_textblob)
                data['VADER Sentiment'] = data['cleaned_text'].apply(analyze_sentiment_vader)

                # Create pie charts for sentiment distribution
                create_pie_chart(data['TextBlob Sentiment'], data['VADER Sentiment'])

                # Generate word clouds for each sentiment
                create_wordcloud(data, 'TextBlob Sentiment')
                create_wordcloud(data, 'VADER Sentiment')

                # Train and evaluate models, display results (Logistic Regression and Naive Bayes)
                results_df = pd.DataFrame(np.zeros((2, 5)), columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC score'])
                results_df.index = ['Logistic Regression (LR)', 'Naïve Bayes Classifier (NB)']

                # Logistic Regression
                lr_model = LogisticRegression()
                lr_model.fit(X_train, y_train)
                y_pred_lr = lr_model.predict(X_test)
                results_df.loc['Logistic Regression (LR)', :] = [
                    accuracy_score(y_test, y_pred_lr),
                    precision_score(y_test, y_pred_lr, average='weighted'),
                    recall_score(y_test, y_pred_lr, average='weighted'),
                    f1_score(y_test, y_pred_lr, average='weighted'),
                    roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr')
                ]

                # Naive Bayes
                nb_model = MultinomialNB()
                nb_model.fit(X_train, y_train)
                y_pred_nb = nb_model.predict(X_test)
                results_df.loc['Naïve Bayes Classifier (NB)', :] = [
                    accuracy_score(y_test, y_pred_nb),
                    precision_score(y_test, y_pred_nb, average='weighted'),
                    recall_score(y_test, y_pred_nb, average='weighted'),
                    f1_score(y_test, y_pred_nb, average='weighted'),
                    roc_auc_score(y_test, nb_model.predict_proba(X_test), multi_class='ovr')
                ]

                # Move model evaluation display to the end
                st.write("### Model Evaluation Results")
                st.table(results_df)

            else:
                st.error("The uploaded CSV file must contain 'text' and 'sentiment' columns.")

def analyze_sentiment_textblob(text):
    """ Analyze the sentiment using TextBlob """
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0.05:
        return "Positive"
    elif sentiment < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_vader(text):
    """ Analyze the sentiment using NLTK VADER """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    main()
