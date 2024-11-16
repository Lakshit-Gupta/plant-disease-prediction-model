import base64
import numpy as np
import streamlit as st
from PIL import ImageOps, Image


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    Classify the image using the provided model and class names.

    Parameters:
        image (PIL Image): The input image to be classified.
        model (tensorflow.keras.Model): The trained model for prediction.
        class_names (dict): A dictionary mapping class indices to class names.

    Returns:
        str: Predicted class name.
        float: Confidence score of the prediction.
    """
    # Convert image to (256, 256) and normalize
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize the image

    # Prepare model input
    data = np.expand_dims(normalized_image_array, axis=0)

    # Make prediction
    prediction = model.predict(data)[0]
    index = np.argmax(prediction)  # Get the index of the predicted class
    class_name = class_names[str(index)]  # Fetch the class name from the class indices
    confidence_score = prediction[index]  # Get the confidence score of the prediction

    return class_name, confidence_score
