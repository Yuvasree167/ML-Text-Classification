
import io
import joblib
import requests

# Raw GitHub URL of the saved model
MODEL_URL = "https://raw.githubusercontent.com/Yuvasree167/ML-Text-Classification/main/20newsgroups_model.joblib"

print("Fetching the trained model directly from GitHub...")

try:
    # Download the model from GitHub
    response = requests.get(MODEL_URL)
    response.raise_for_status()

    # Load the model from memory
    model_bytes = io.BytesIO(response.content)
    model = joblib.load(model_bytes)

    print("Model loaded successfully into memory!")

    # New unseen text samples
    new_samples = [
        "The local team won the championship game in overtime last night.",
        "Scientists have discovered a new planet orbiting a distant star.",
        "My computer graphics card is overheating when I try to boot up the operating system."
    ]

    # Predict categories
    predictions = model.predict(new_samples)

    print("\n----- Prediction Results -----")

    for text, prediction in zip(new_samples, predictions):
        print("Text:", text)
        print("Predicted Category ID:", prediction)
        print()

except requests.exceptions.RequestException as e:
    print(f"Error fetching model from GitHub: {e}")

except Exception as e:
    print(f"An error occurred while loading or running the model: {e}")
