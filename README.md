# Cat vs. Dog Image Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app) <!-- Replace with your deployed app URL -->

## Overview

This project is an end-to-end deep learning application that classifies images as either a cat or a dog. It features a convolutional neural network (CNN) model built with TensorFlow/Keras and is deployed as an interactive web application using Streamlit.

The model is trained on the popular Kaggle "Dogs vs. Cats" dataset. To ensure the GitHub repository remains lightweight and scalable, the large model file (170MB) is hosted on Google Drive and downloaded dynamically by the Streamlit application on its initial run.

## Features

- **Deep Learning Model:** A robust CNN architecture for accurate binary image classification.
- **Interactive UI:** A clean and user-friendly web interface built with Streamlit that allows users to easily upload images.
- **Dynamic Model Fetching:** The application intelligently downloads the trained model from Google Drive, avoiding large file storage in the Git repository.
- **Real-time Prediction:** Displays the predicted class (Cat or Dog) along with the model's confidence score.
- **Professional Project Structure:** Follows best practices for project organization, dependency management, and version control.

## Tech Stack

- **Backend & Model:** Python, TensorFlow, Keras
- **Frontend:** Streamlit
- **Data Handling:** NumPy, Pillow
- **Model Hosting:** Google Drive
- **Version Control:** Git & GitHub

## Project Structure

```
Cat-dog-classification/
├── .gitignore
├── app.py
├── cats_v_dogs_classification.ipynb
├── README.md
└── requirements.txt
```
- **`app.py`**: The main script that runs the Streamlit web application.
- **`requirements.txt`**: A list of all Python dependencies required for the project.
- **`cats_v_dogs_classification.ipynb`**: The Jupyter Notebook containing the end-to-end process for training and evaluating the CNN model.
- **`.gitignore`**: Specifies which files and directories to exclude from version control.
- **`README.md`**: This file.

## Setup and Local Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 - 3.11
- Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

## Model Training

The CNN model was trained using the `cats_v_dogs_classification.ipynb` notebook. The notebook covers all steps from data acquisition and preprocessing to model building, training, and evaluation. The final trained model, `my_model.keras`, is then hosted on Google Drive for the application to use.

## Deployment

This application is designed for easy deployment on Streamlit Cloud.

1.  **Push to GitHub:** All code is pushed to this GitHub repository.
2.  **Streamlit Cloud:** The repository is linked to Streamlit Cloud, which automatically deploys the application. The first time the app starts, it downloads the `.keras` model file from the configured Google Drive link, making the deployment process seamless and efficient.
