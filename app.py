import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .st-bk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_keras_model():
    """
    Downloads the model from the Hugging Face Hub and caches it.
    Includes robust error handling.
    """
    # --- Configuration ---
    # The repository ID of your model on the Hugging Face Hub.
    REPO_ID = "noman786/cat-dog-classifier"
    # The name of the model file in the repository.
    FILENAME = "cat_dog_cls_model.keras"

    try:
        # Download the model from the Hub
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        # Load the model
        # Using compile=False can be faster if you only need to do inference.
        model = load_model(model_path, compile=False)
        return model
    
    except Exception as e:
        # If any error occurs, display a user-friendly message.
        st.error(f"An error occurred while loading the model: {e}")
        st.error(
            "This could be due to a network issue or a problem with the model file on the Hub. "
            "Please check the repository link and try again."
        )
        st.info(f"Model repository: https://huggingface.co/{REPO_ID}")
        return None

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the uploaded image to fit the model's input requirements.
    """
    # Resize the image to the target size
    img = image.resize((256, 256))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the pixel values to the [0, 1] range
    img_array = img_array / 255.0
    # Expand the dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main Application ---
def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title("🐾 Cat vs. Dog Image Classifier")
    st.markdown(
        "Upload an image of a cat or a dog, and the model will predict which one it is!"
    )

    # Load the model
    model = load_keras_model()

    # If the model fails to load, stop the app
    if model is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        st.header("Or Use Webcam")
        
        # Initialize session state for webcam activation
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False

        if st.button("Activate Webcam"):
            st.session_state.webcam_active = True

        camera_image = None
        if st.session_state.webcam_active:
            camera_image = st.camera_input("Take a picture")
            if camera_image: # If an image is taken, deactivate webcam to prevent continuous capture
                st.session_state.webcam_active = False

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = uploaded_file
    elif camera_image is not None:
        image_to_process = camera_image

    if image_to_process is not None:
        try:
            image = Image.open(image_to_process)
            
            with col1:
                st.image(image, caption="Your Image", use_container_width=True)

            # Preprocess the image and make a prediction
            with st.spinner("Analyzing the image..."):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                confidence = prediction[0][0]

            with col2:
                st.header("Prediction")
                if confidence > 0.5:
                    st.markdown(f"## This is a Dog! 🐶")
                    st.progress(float(confidence))
                    st.write(f"**Confidence:** {confidence:.2%}")
                else:
                    st.markdown(f"## This is a Cat! 🐱")
                    st.progress(float(1 - confidence))
                    st.write(f"**Confidence:** {(1 - confidence):.2%}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        with col2:
            st.info("Please upload an image or use the webcam to see the prediction.")

if __name__ == "__main__":
    main()
