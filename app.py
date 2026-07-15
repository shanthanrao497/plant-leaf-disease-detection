from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'model/ensemble_model.h5'
ensemble_model = load_model(MODEL_PATH)
ensemble_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load class names
DATASET_DIR = "model/dataset"
class_names = sorted(os.listdir(DATASET_DIR)) if os.path.exists(DATASET_DIR) else []

def preprocess_image(image_path):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        image = preprocess_image(file_path)
        prediction = ensemble_model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index] if class_names else 'Unknown'
        
        return jsonify({'class': predicted_class, 'image_url': file_path})

if __name__ == '__main__':
    app.run(debug=True)
