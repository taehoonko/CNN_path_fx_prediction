import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('my_densenet121_model.h5')

# Load the new test image
img_path = '/path/to/test/image.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the input image
img_array = tf.keras.applications.densenet.preprocess_input(img_array)

# Make predictions on the input image
predictions = model.predict(img_array)

# Get the predicted class label
if predictions >= 0.5:
    predicted_class = 1
else:
    predicted_class = 0

# Print the predicted class label
print('The predicted class is: ', predicted_class)
print('Confidence: ', predictions)