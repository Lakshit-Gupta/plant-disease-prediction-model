import os
import random
import numpy as np
import json
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from util import classify, set_background  # Assuming you still want to use the 'classify' and 'set_background' functions
#setting seed for reproducibly
import random
random.seed(69)
import numpy as np
np.random.seed(69)
import tensorflow as tf
tf.random.set_seed(69)
# Set background image (optional)
set_background('img.png')

# Set the title and header of the Streamlit app
st.title('Plant Disease Classification')
st.header('Please upload an image to check if your plant has a disease.')

# Upload file (for image)
file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

# Load the model and class indices (from JSON)
model = load_model('plant_disease_prediction_model_1.keras') #CHANGED THE LAST DIGIT AS YOU WANT TO CHOOSE THE MODEL
with open('class_indices.json') as f:
    class_indices = json.load(f)


# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = image.convert('RGB')  # Ensure all images are RGB
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Scale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Accessing the predicted class name from the class indices
    try:
        predicted_class_name = class_indices[str(predicted_class_index)]
    except KeyError:
        st.write(f"KeyError: {predicted_class_index} not found in class indices.")
        return None, predictions[0]

    confidence_scores = predictions[0]
    return predicted_class_name, confidence_scores


# Display the uploaded image and the prediction only after clicking the button
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Add Predict button
    if st.button('Predict'):
        # Predict the class of the uploaded image
        predicted_class_name, confidence_scores = predict_image_class(model, image, class_indices)

        # Output the result if prediction was successful
        if predicted_class_name:
            # Create a black/white box around the confidence score
            sorted_confidence_scores = sorted(zip(class_indices.values(), confidence_scores), key=lambda x: x[1], reverse=True)

            # Combine all the output into one formatted string
            confidence_str = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n".join([f"{class_name}: {score:.6f}" for class_name, score in sorted_confidence_scores])

            st.markdown(f"""
                <div style="background-color: black; padding: 20px; border-radius: 10px;">
                    <h2 style="color: white;">Predicted Class: {predicted_class_name}</h2>
                    <h3 style="color: white;">Confidence: {confidence_scores[np.argmax(confidence_scores)] * 100:.2f}%</h3>
                    <h3 style="color: white;">Confidence Scores (from highest to lowest):</h3>
                    <pre style="color: white;">{confidence_str}</pre>
                </div>
            """, unsafe_allow_html=True)

