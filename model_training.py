"""model_training.py
Clean training script for E-commerce Sentiment Analysis (MultinomialNB)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['Score'] != 3].copy()
    df['label'] = (df['Score'] >= 4).astype(int)
    return df

def train(path_csv):
    df = load_data(path_csv)
    X = df['Text'].astype(str).fillna("")
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    vect = CountVectorizer(stop_words='english', max_features=20000)
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, "model_nb.joblib")
    joblib.dump(vect, "vectorizer.joblib")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_training.py data/Reviews.csv")
    else:
        train(sys.argv[1])
