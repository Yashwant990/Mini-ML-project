import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

DATA_PATH = os.path.join("data", "training_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    return df["text"].tolist(), df["label"].tolist()

def train_model():
    texts, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print("Model saved at", MODEL_PATH)

if __name__ == "__main__":
    train_model()
