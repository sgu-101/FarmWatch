# FarmWatch

FarmWatch is a web application developed with Flask that integrates a chatbot and convolutional neural network (CNN) models to classify pests and plant diseases.

# Demo

https://youtu.be/qRbl7Mcc1oQ

## Features

- **Chatbot Integration**: Ask questions about agricultural practices and get expert advice through the Agri-Pilot chatbot.
- **Pest and Disease Classification**: Upload images to classify pests and plant diseases using CNN models.
- **User-Friendly Interface**: A web-based interface for easy interaction with the models and chatbot.

## Project Structure

- **`models`**: Contains the code used to create and train the CNN models.
- **`static`**: For static files such as images used on the website.
- **`templates`**: Holds the HTML files for the web pages.
- **`uploads`**: Stores images uploaded by users for classification.
- **`disease_names.json`**: JSON file containing the class names for plant diseases.
- **`pest_names.json`**: JSON file containing the class names for pests.
- **`pest_detection_model.keras`**: Trained model for pest detection.
- **`plant_disease_detection.keras`**: Trained model for plant disease detection.
- **`app.py`**: The web application in question using Flask 

## How to Run Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/FarmWatch.git
   cd FarmWatch
   
2. **Install Dependencies**

3. **Run the Flask Application in Development Mode(Mac)**
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
3. **Run the Flask Application in Development Mode(Windows)**
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run

## Acknowledgements

Flask: For the web framework.

TensorFlow/Keras: For the CNN models.

LangChain: For the chatbot integration.

Kevin Walsh: For being our super cool mentor




