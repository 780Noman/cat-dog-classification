# 🐾 Cat vs. Dog Image Classifier

This is a Streamlit web application that leverages a pre-trained Convolutional Neural Network (CNN) model to classify images as either a cat or a dog. Users can upload an image or use their webcam to get real-time predictions. The model is hosted and loaded directly from the Hugging Face Hub, ensuring efficient deployment and model management.

## ✨ Features

*   **Image Upload:** Easily upload `.jpg`, `.jpeg`, or `.png` images for classification.
*   **Webcam Integration:** Use your device's camera to capture an image and get an instant prediction (webcam activates on button click).
*   **Real-time Prediction:** Fast and accurate classification of cat and dog images.
*   **Confidence Score:** Displays the model's confidence in its prediction.
*   **Hugging Face Model Hub Integration:** Model is dynamically loaded from `noman786/cat-dog-classifier` on Hugging Face Hub.
*   **Responsive UI:** Built with Streamlit for a clean and interactive user experience.

## 🚀 Technologies Used

*   **Python**
*   **Streamlit:** For building the interactive web application.
*   **TensorFlow/Keras:** For the deep learning model.
*   **PIL (Pillow):** For image processing.
*   **Hugging Face Hub (huggingface_hub):** For model hosting and loading.
*   **NumPy:** For numerical operations.

## ⚙️ Installation and Setup

Follow these steps to get a local copy of the project up and running on your machine.

### Prerequisites

*   Python 3.8+
*   Git

### 1. Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/your-username/Cat-dog-classification.git
cd Cat-dog-classification
```

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to create a virtual environment to manage project dependencies:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

*   **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4. Install Dependencies

Once your virtual environment is active, install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🏃 How to Run the Application

After completing the installation steps, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your default web browser. If it doesn't open automatically, copy and paste the provided local URL (e.g., `http://localhost:8501`) into your browser.

## 🧠 Model Information

The core of this application is a deep learning model trained to distinguish between cats and dogs. This model is stored on the Hugging Face Model Hub under the repository `noman786/cat-dog-classifier`. The `app.py` script automatically downloads and loads this model when the application starts, ensuring you always use the latest version without needing to manage large model files locally.

## ☁️ Deployment

### Deploying to Streamlit Cloud (Recommended)

Streamlit Cloud provides the easiest way to deploy your Streamlit applications directly from a GitHub repository.

1.  **Push to GitHub:** Ensure your project is pushed to a GitHub repository (as you are planning to do).
2.  **Sign in to Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3.  **Deploy an App:** Click on "New app" or "Deploy an app".
4.  **Connect Repository:** Select your `Cat-dog-classification` repository and the `main` branch.
5.  **Set Main File Path:** Ensure the "Main file path" is set to `app.py`.
6.  **Advanced Settings (Optional):** If your app requires specific Python versions or other configurations, you can set them here.
7.  **Deploy!** Click "Deploy!" and Streamlit Cloud will build and deploy your application.

### Deploying to Hugging Face Spaces

Given that your model is already on Hugging Face Hub, deploying the Streamlit app to Hugging Face Spaces is another excellent option.

1.  **Create a New Space:** Go to [huggingface.co/spaces/new](https://huggingface.co/spaces/new) and create a new Space.
2.  **Choose SDK:** Select "Streamlit" as the SDK.
3.  **Link to GitHub (Optional but Recommended):** You can link your Space directly to your GitHub repository. This allows automatic updates when you push changes to GitHub.
4.  **`requirements.txt`:** Ensure your `requirements.txt` file is in the root of your repository. Hugging Face Spaces will automatically install these dependencies.
5.  **`app.py`:** Your main Streamlit application file (`app.py`) should also be in the root.
6.  **Push Code:** Push your entire project (including `app.py` and `requirements.txt`) to your Hugging Face Space's Git repository.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you plan to add one).

## 📧 Contact

Noman (Your Name/Alias) - [your.email@example.com](mailto:your.email@example.com)
Project Link: [https://github.com/your-username/Cat-dog-classification](https://github.com/your-username/Cat-dog-classification)
