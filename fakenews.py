# importing what I need
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
import matplotlib.pyplot as plt
import re
from collections import Counter
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# (0 = fake, 1 = real)- kaggle

# load datasets
df = pd.read_csv("/Users/arisaenokido/Desktop/PYTHON/API MAJOR TEST/WELFake_Dataset.csv")

y = df["label"]

#for streamilt
st.title("Arisa's Incredibly Accurate (...) Fake News Classifier!")
if st.checkbox("Check this to show the original data!!"):
    
    st.write(df.head())
st.subheader("quite decent model accuracy if I say so myself...")
stop_words = set(stopwords.words('english'))

#stemmmmmmmmmmm the headline 
lemmatizer = WordNetLemmatizer()

def clean_title(title):
    title = str(title).lower()
    title = re.sub(r'[^a-zA-Z\s]', '', title)
    words = word_tokenize(title)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)



df["cleaned_title"] = df["title"].apply(clean_title)

#vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_title"])

#train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
#pred and accuracy
y_pred = nb_classifier.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
st.info(f"Model Accuracy: {score:.2f}")

#user input interaction
user_input = st.text_input("input news headline here and click enter: ")
if user_input:
    cleaned_input = clean_title(user_input)
    vectorize_input = vectorizer.transform([cleaned_input])
    prediction = nb_classifier.predict(vectorize_input)[0]
    probability = nb_classifier.predict_proba(vectorize_input)[0]
    confidence = max(probability) * 100
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    st.write(f"Prediction: {label}!!!!! ")
    st.write(f"Confidence: {confidence:.2f}%!!!")


# fake real dist
st.header("Fake News vs Real News Count")
fig, ax1 = plt.subplots()
df["label"].value_counts().plot(kind="bar", color=["rosybrown", "slategrey"], ax=ax1)
ax1.set_xticklabels(["Fake News", "Real News"])
ax1.set_ylabel("Count")
st.pyplot(fig)

# dist pie chart
class_counts = df['label'].value_counts()
sizes = class_counts.values
st.subheader("Distribution of Fake vs. Real News")
st.markdown("This one shows the percentage of distribution of fake and real news in my dataset!!")
fig1, ax = plt.subplots()
ax.pie(sizes, labels=['Fake News', 'Real News'], colors=['lightcoral','cornflowerblue'], autopct='%1.1f%%', startangle=90)
st.pyplot(fig1)
