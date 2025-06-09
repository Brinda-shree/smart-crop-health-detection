import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("crop_health_model.h5")

# Image path (replace with actual test image path)
img_path = "test.jpg"

# Preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)

# Class names (based on your dataset folder names)
class_names = ['Potato___Early_blight', 'Tomato_healthy', 'Tomato_Late_blight']

# Get predicted class and confidence
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

# Output
print(f"Predicted: {predicted_class} with confidence {confidence:.2f}%")
