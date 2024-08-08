import os
import json
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = load_model('pest_detection_model.keras')

# Load class labels from JSON file
with open('class_names.json', 'r') as f:
    CLASS_LABELS = json.load(f)

def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Adjust target size as per your model input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = prepare_image(file_path)
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = CLASS_LABELS[predicted_class_index]
        return render_template('index.html', prediction_text=f'Predicted class: {predicted_class_label}')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
