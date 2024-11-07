from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from PIL import Image
import cv2
import numpy as np

from utils.processing import preprocess_image
from utils.classifier import classify_image
from utils.detector import run_inference_tflite

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No image part')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open image with OpenCV
        image = cv2.imread(filepath)
        model_path = 'models/unisign_square_feature192_05122023.tflite'
        # Run image through the preprocessing and classification pipeline
        # preprocessed_image = preprocess_image(image)
        # classification_result = classify_image(preprocessed_image)
        detection_result = run_inference_tflite(filepath, model_path)
        #detection_result = str(detection_result)
        return render_template(
            'index.html',
            filename=filename,
           # classification_result=classification_result,
            detection_result=detection_result
        )
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
