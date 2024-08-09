import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

directory = 'pestdetection'

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len([3,3,3,3]), activation='softmax')  # Use softmax for multi-class classification
])


home_dir = os.path.expanduser('~')  # Use the user's home directory
save_dir = os.path.join(home_dir, 'downloads', 'futuremaker')
os.makedirs(save_dir, exist_ok=True)

# Save the model
model_path = os.path.join(save_dir, 'pest_detection_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")