# from flask import Flask, jsonify, request
# from werkzeug.utils import secure_filename
# import numpy as np
# import tensorflow as tf
# import cv2
# import json
# import os
# import tempfile
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Load the model
# cnn = tf.keras.models.load_model("new_model_4.h5")
# logging.info("Model loaded successfully.")

# # Define a function to preprocess the input image
# def preprocess_image(image):
#     image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
#     image = np.expand_dims(image, axis=0)
#     image = image / 255.0
#     return image

# # Define a route for the API
# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided."}), 400

#     image_file = request.files["image"]

#     if image_file.filename == '':
#         return jsonify({"error": "No image selected for uploading."}), 400

#     filename = secure_filename(image_file.filename)
#     if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         return jsonify({"error": "Invalid file extension."}), 400

#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         image_file.save(temp_file.name)
#         image = cv2.imread(temp_file.name)
#         image = preprocess_image(image)

#     predictions = cnn.predict(image)
#     logging.info("Prediction values: %s", predictions[0].tolist())

#     tolerance = 1e-8  # Adjust tolerance as needed (smaller for stricter comparison)
#     highest_value = np.max(predictions[0])
#     filtered_indices = np.where(np.abs(predictions[0] - highest_value) < tolerance)[0]

#     # If a value is within tolerance of the maximum, choose its index
#     if len(filtered_indices) > 0:
#         result_index = filtered_indices[0]
#     else:
#         # Fallback to default behavior (argmax) if no value within tolerance
#         result_index = np.argmax(predictions[0])

#     logging.info("Result index: %d", result_index)

#     with open("class_names.json") as f:
#         class_names = json.load(f)
#     result_class = class_names[result_index]

#     os.unlink(temp_file.name)
#     return jsonify({"class": result_class})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the model
cnn = tf.keras.models.load_model("new_model_2.h5")

# Load class names from JSON file
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def load_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((64, 64))
    image_array = np.array(image)
    image_array = np.array([image_array])  # Convert single image to batch
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_data = file.read()
            image_array = load_image(image_data)
            predictions = cnn.predict(image_array)
            result_index = np.where(predictions[0] == max(predictions[0]))
            predicted_class = class_names[result_index[0][0]]
            return jsonify({"predicted_class": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
