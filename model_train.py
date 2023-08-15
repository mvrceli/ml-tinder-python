import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from sklearn.linear_model import LogisticRegression
import pickle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

hair_colors = ['black', 'blond', 'red', 'brown']

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match VGG16 input shape
    image = image.astype("float") / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = base_model.predict(image)
    features = features.flatten()  # Flatten the feature vector
    return features

# Specify the directory containing the labeled images
data_directory = 'labeled_images_directory_path'

# Load pre-trained VGG16 model (exclude top fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False)

# Extract features from labeled images and collect them into X and y lists
X = []
y = []

for hair_color in hair_colors:
    folder_path = os.path.join(data_directory, hair_color)
    print(folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            X.append(features)
            y.append(hair_color)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train the classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, y)

# Save the trained model using pickle
model_path = 'hair_color_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(classifier, file)