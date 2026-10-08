
import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

print("Loading 20 Newsgroups dataset...")

train = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

print("Building the Multinomial Naive Bayes model pipeline...")

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

print("Training the model...")

model.fit(train.data, train.target)

predicted = model.predict(test.data)

accuracy = accuracy_score(test.target, predicted)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        test.target,
        predicted,
        target_names=train.target_names
    )
)

model_filename = "20newsgroups_model.joblib"

joblib.dump(
    model,
    model_filename,
    compress=3
)

print(f"Model saved successfully as {model_filename}")
