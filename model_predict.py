import cv2
import numpy as np
import pickle
from tensorflow.keras.applications import VGG16


model_path = 'hair_color_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

base_model = VGG16(weights='imagenet', include_top=False)

def extract_features(screenshot_path):
    image = cv2.imread(screenshot_path)
    image = cv2.resize(image, (224, 224))  
    image = image.astype("float") / 255.0  
    image = np.expand_dims(image, axis=0)  
    features = base_model.predict(image)
    features = features.flatten()  
    return features

def predict_hair_color(screenshot_path):
    features = extract_features(screenshot_path)

    predictions = model.predict([features])
    global predicted_color
    predicted_color = predictions[0]
    
    print(f"Predicted Hair Color: {predicted_color}")
    return predicted_color






