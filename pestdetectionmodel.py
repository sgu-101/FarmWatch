import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

img_size = 128

def clean_class_name(class_name):
    # Remove '_train' or '_test' suffix
    if class_name.endswith('_train'):
        return class_name[:-6]
    elif class_name.endswith('_test'):
        return class_name[:-5]
    return class_name

def load_images_from_folders(base_folder, img_size):
    images = []
    labels = []
    class_names = os.listdir(base_folder)
    
    for class_name in class_names:
        class_folder = os.path.join(base_folder, class_name)
        if os.path.isdir(class_folder):
            for file in os.listdir(class_folder):
                if file.startswith('.'):  # Ignore hidden files
                    continue
                file_path = os.path.join(class_folder, file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    try:
                        img = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB mode
                        img = img.resize((img_size, img_size))
                        images.append(np.array(img))
                        labels.append(clean_class_name(class_name))  # Clean class name
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    return images, labels, list(set(clean_class_name(cn) for cn in class_names))

def display_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis("off")

# Set image size and directories
img_size = 128
train_folder = 'pestdetection/train'
val_folder = 'pestdetection/test'

# Load training images and labels
train_images, train_labels, train_class_names = load_images_from_folders(train_folder, img_size)
print(f"Number of training images loaded: {len(train_images)}")
print(f"Number of training labels loaded: {len(train_labels)}")

# Load validation images and labels
val_images, val_labels, val_class_names = load_images_from_folders(val_folder, img_size)
print(f"Number of validation images loaded: {len(val_images)}")
print(f"Number of validation labels loaded: {len(val_labels)}")

# Combine class names from both training and validation sets
all_class_names = list(set(train_class_names + val_class_names))
print(f"All class names: {all_class_names}")

# Normalize pixel values
train_images = train_images / 255.0
val_images = val_images / 255.0

# Encode labels using the combined class names
le = LabelEncoder()
le.fit(all_class_names)
train_labels_encoded = le.transform(train_labels)
val_labels_encoded = le.transform(val_labels)

# Build the model
model = Sequential([
    Input(shape=(img_size, img_size, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(all_class_names), activation='softmax')  # Use softmax for multi-class classification
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels_encoded, epochs=15, validation_data=(val_images, val_labels_encoded), batch_size=32)

# Save the model and class names
home_dir = os.path.expanduser('~')  # Use the user's home directory
save_dir = os.path.join(home_dir, 'downloads', 'futuremaker')
os.makedirs(save_dir, exist_ok=True)

# Save the model
model_path = os.path.join(save_dir, 'pest_detection_model.keras')
model.save(model_path)
print(f"Model saved to {model_path}")

# Save class names
class_names_file = os.path.join(save_dir, 'class_names.json')
with open(class_names_file, 'w') as f:
    json.dump(all_class_names, f)
print(f"Class names saved to {class_names_file}")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
