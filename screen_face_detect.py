import cv2
import numpy as np
import pyautogui
import time

counter = 0

def screenshot():
    time.sleep(2)
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    return screenshot

def detect_faces(screenshot_path):
    image = cv2.imread(screenshot_path)
    counter = 0
    face_recog = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_recog.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        
        expanded_w = 2 * w
        expanded_h = 2 * h

        new_x = max(x - (expanded_w - w) // 2, 0)
        new_y = max(y - (expanded_h - h) // 2, 0)

        face_roi = image[new_y:new_y+expanded_h, new_x:new_x+expanded_w]


        cv2.imwrite(f"face_{counter}_.png", face_roi)
        counter += 1




