from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import cv2
import json
import os
import tempfile
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
cnn = tf.keras.models.load_model("new_model_1.h5")
logging.info("Model loaded successfully.")

# Define a function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Define a route for the API
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400

    image_file = request.files["image"]

    if image_file.filename == '':
        return jsonify({"error": "No image selected for uploading."}), 400

    filename = secure_filename(image_file.filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file extension."}), 400

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        image_file.save(temp_file.name)
        image = cv2.imread(temp_file.name)
        image = preprocess_image(image)

    predictions = cnn.predict(image)
    result_index = np.argmax(predictions[0])

    with open("class_names.json") as f:
        class_names = json.load(f)
    result_class = class_names[result_index]

    os.unlink(temp_file.name)
    return jsonify({"class": result_class})

if __name__ == "__main__":
    app.run(debug=True)
