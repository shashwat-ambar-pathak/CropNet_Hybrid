import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from preventions import prevention_tips
from class_labels import class_names

# ----------------- Load Model -----------------
model = load_model(
    "best_cropnet_model.keras",
    custom_objects={"top3_acc": TopKCategoricalAccuracy(k=3)}
)

# ----------------- Prediction Function -----------------
def predict_disease(img_input, top_k=3):

    # If input is a file path
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            return "Image not found.", 0, ["No data available."], []

        img = image.load_img(img_input, target_size=(224, 224))

    # If input is a PIL Image
    elif isinstance(img_input, Image.Image):
        img = img_input.resize((224, 224))

    else:
        raise ValueError("Input must be a file path or PIL Image.")

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    top_indices = prediction.argsort()[-top_k:][::-1]

    top_predictions = [
        (class_names[i], float(prediction[i] * 100))
        for i in top_indices
    ]

    predicted_class, confidence = top_predictions[0]

    preventions = prevention_tips.get(
        predicted_class,
        ["No data available."]
    )

    return predicted_class, confidence, preventions, top_predictions
