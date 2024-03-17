from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import cv2
import json
import os
import uuid

app = Flask(__name__)

# Load the model
cnn = tf.keras.models.load_model("./trained_model_best.h5")

# Define a function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# # Load the validation dataset
# validation_set = tf.keras.utils.image_dataset_from_directory(
#     './archive/validation',
#     labels="inferred",
#     label_mode="categorical",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(64,64),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False
# )

# # Get the class names
# class_names = validation_set.class_names

# Define a route for the API
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    image = request.files["image"]

    # Generate a unique file name
    file_name = f"{uuid.uuid4()}.jpg"

    # Save the image to disk
    image.save(file_name)

    # Preprocess the image
    image_path = os.path.join(os.getcwd(), file_name)
    image = preprocess_image(image_path)

    # Make a prediction
    predictions = cnn.predict(image)

    # Get the index of the highest prediction
    result_index = np.argmax(predictions[0])

    # # Get the class name
    # result_class = class_names[result_index]

    with open("./class_names.json") as f:
        class_names = json.load(f)
    result_class = class_names[result_index]

    # Delete the temporary file
    os.remove(image_path)

    # Return the result
    return jsonify({"class": result_class})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)