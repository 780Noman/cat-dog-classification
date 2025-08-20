
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import os
from tqdm import tqdm

# Set page config
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
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
    .st-emotion-cache-1v0mbdj > img{
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Model Downloading and Loading ---
def download_file_from_google_drive(id, destination):
    URL = f'https://drive.google.com/uc?export=download&id={id}'
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        st.error("An error occurred during file download.")
        return False
    return True

@st.cache_resource
def load_keras_model():
    """
    Downloads the model from Google Drive if not present, then loads it.
    The `st.cache_resource` decorator ensures the model is loaded only once.
    """
    MODEL_PATH = "my_model.keras"
    FILE_ID = "1M-HNEJqbz6PzjhX6WHHKLPbjZpPRWLjP" # Replace with your file ID

    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally. Downloading from Google Drive... (this may take a moment)")
        download_file_from_google_drive(FILE_ID, MODEL_PATH)
        st.success("Model downloaded successfully!")

    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the Google Drive File ID is correct and the file is accessible.")
        return None

model = load_keras_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to fit the model's input requirements.
    - Resizes to (256, 256)
    - Converts to a NumPy array
    - Normalizes pixel values
    - Expands dimensions for the model
    """
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- UI Layout ---
st.title("🐾 Cat vs. Dog Image Classifier")
st.markdown(
    "Upload an image of a cat or a dog, and the model will predict which one it is!"
)

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    confidence = prediction[0][0]

    with col2:
        st.header("Prediction")
        if confidence > 0.5:
            st.markdown(
                f"## This is a Dog! 🐶"
            )
            st.progress(confidence)
            st.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.markdown(
                f"## This is a Cat! 🐱"
            )
            st.progress(1-confidence)
            st.write(f"**Confidence:** {1-confidence:.2f}")
else:
    with col2:
        st.info("Please upload an image to see the prediction.")

