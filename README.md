
# Text Classification using Multinomial Naive Bayes

This project performs text classification using the scikit-learn
20 Newsgroups dataset.

## Machine Learning Model

The model uses:

- TF-IDF Vectorizer
- Multinomial Naive Bayes

## Files

- train.py - Trains and evaluates the model
- requirements.txt - Required Python packages
- 20newsgroups_model.joblib - Saved trained ML model
- README.md - Project documentation

## Model Reusability

The trained model is saved using joblib and can be loaded later
without retraining the model.

The model can also be downloaded from GitHub and reused for
predicting new text samples.
