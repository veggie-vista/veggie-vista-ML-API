web: gunicorn app:app --bind 0.0.0.0:$PORT

# Assuming your model file is named "trained_model_best.h5"
# and your class names file is named "class_names.json"

# Set environment variables to specify their locations
heroku config:set MODEL_PATH=./trained_model_best.h5
heroku config:set CLASS_NAMES_PATH=./class_names.json

# Load the model and class names in your app.py file
# using the environment variables set above
import os

model_path = os.environ.get('MODEL_PATH')
class_names_path = os.environ.get('CLASS_NAMES_PATH')

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(class_names_path) as f:
    class_names = json.load(f)