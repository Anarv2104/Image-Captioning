# 🖼️ Image Caption Generator

This Flask application generates captions for uploaded images using a pre-trained EfficientNetB0 model. The application uses TensorFlow and Keras for model handling and image processing.

## ✨ Features

- 📷 Upload an image and receive a generated caption.
- 🧠 Uses a pre-trained EfficientNetB0 model for image feature extraction.
- 💻 Simple web interface for image uploading and caption display.

## 🛠️ Requirements

- Flask
- NumPy
- Pillow
- TensorFlow
- TensorFlow Hub
- TensorFlow Text
- OpenCV (headless version)

## 📦 Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `static/uploads` directory exists:
   ```bash
   mkdir -p static/uploads
   ```

## 🚀 Usage

1. Run the Flask application:
   ```bash
   flask run
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Upload an image using the provided form, and receive a generated caption for your image.

## 📂 Project Structure

- `app.py`: The main Flask application file.
- `templates/index.html`: The HTML template for the web interface.
- `static/uploads`: Directory to store uploaded images.

## 🙏 Acknowledgments

- The application uses EfficientNetB0 from TensorFlow for image feature extraction.
- The project setup and structure are based on Flask web development best practices.
