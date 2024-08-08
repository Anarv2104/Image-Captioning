from flask import Flask, request, render_template, url_for
import os
import uuid
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load pre-trained image captioning model
cnn_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(299, 299, 3)),
    cnn_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Flatten the output
])
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1280,)),
    tf.keras.layers.Dense(1280, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax'),
])

# Load vocabulary
vocab = [
    '<start>', '<end>', 'a', 'an', 'the', 'man', 'woman', 'child', 'dog', 'cat', 'bird', 
    'car', 'bicycle', 'on', 'in', 'with', 'and', 'is', 'are', 'running', 'jumping', 
    'playing', 'sitting', 'standing', 'walking', 'eating', 'drinking', 'looking',
]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Save the uploaded image
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        img = load_img(filepath, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        features = encoder.predict(img_array)

        # Ensure features have the correct shape
        if features.shape[-1] != 1280:
            raise ValueError(f"Expected encoder output shape to have 1280 features, but got {features.shape[-1]}")
 
        # Generate caption with top-k sampling
        caption = ''
        tokenized_caption = np.array([[vocab.index('<start>')]])
        for i in range(50):  # generate caption of length 50
            output = decoder.predict([features, tokenized_caption])
            output = output[0, :]  # Corrected indexing to handle 2-dimensional array

            # Top-k sampling
            k = 5
            top_k_indices = np.argsort(output)[-k:]
            top_k_probs = output[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)  # Normalize probabilities
            sampled_token_index = np.random.choice(top_k_indices, p=top_k_probs)
            
            sampled_token = vocab[sampled_token_index]
            if sampled_token == '<end>':
                break
            caption += ' ' + sampled_token
            tokenized_caption = np.append(tokenized_caption, [[sampled_token_index]], axis=1)

# Pass the image URL and caption to the template
        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('index.html', image_url=image_url, caption=caption.strip())
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)