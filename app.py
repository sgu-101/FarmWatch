import os
import json
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained models and class names (you already have this set up)
pest_model = load_model('pest_detection_model.keras')
disease_model = load_model('plant_disease_detection.keras')

with open('pest_names.json', 'r') as f:
    pest_classes = json.load(f)

with open('disease_names.json', 'r') as f:
    disease_classes = json.load(f)

api_key = "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"

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are Agri-Pilot, an expert in the relevant agricultural field a seasoned farmer and agriculture specialist 
    with decades of experience in managing crops, identifying pests, and diagnosing plant diseases. Your extensive 
    knowledge in agriculture makes you an invaluable resource for providing practical solutions and preventive 
    measures for maintaining healthy crops.
    """),
    ("user", "{input}")
])

llm = ChatOpenAI(openai_api_key=api_key)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

def prepare_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route for pest and disease detection
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model_type']
    file = request.files['file']

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if model_type == 'pest':
        target_size = (128, 128)
    elif model_type == 'disease':
        target_size = (256, 256)
    else:
        return "Invalid model type selected."

    img = prepare_image(file_path, target_size)

    if model_type == 'pest':
        prediction = pest_model.predict(img)
        predicted_class = pest_classes[np.argmax(prediction)]
    elif model_type == 'disease':
        prediction = disease_model.predict(img)
        predicted_class = disease_classes[np.argmax(prediction)]

    # Pass prediction_text to the template
    return render_template('index.html', prediction_text=f'Predicted class: {predicted_class}')



# Define a new route for Agri-Pilot interactions
@app.route('/agripilot', methods=['POST'])
def agripilot():
    user_input = request.form['user_input']  # Get input from user via form submission
    response = chain.invoke({"input": user_input})  # Get response from Agri-Pilot
    return jsonify({"response": response})  # Return the response as JSON


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
