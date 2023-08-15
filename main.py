# from model_train import extract_features
import model_predict
import screen_face_detect
import os
from pynput import keyboard
from pynput.keyboard import Key
import pynput
#VARIABLES
face_path = "face_0_.png"
screenshot_path = "screenshot.png"
counter = 0

#MAIN FUNCTIONS
def face_counter():
    counter = 0
    for filename in os.listdir():
        if filename.endswith(".png"):
            counter += 1
    return counter-1


def del_pics():
    for filename in os.listdir():
        if filename.endswith("_.png"):
            file_path = os.path.join(filename)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")


def swipe(predicted_color):
    keyboard = pynput.keyboard.Controller()
    if color_pref == predicted_color:
        keyboard.press(Key.right)
        keyboard.release(Key.right)
    else:
        keyboard.press(Key.left)
        keyboard.release(Key.left)

def on_press(key):
    if key == keyboard.Key.space:
        return False 
    else:
        pass
listener = keyboard.Listener(on_press=on_press)


#ASK FOR COLOR PREFERENCE
color_pref = input("What hair color do you like? (Brown, Black, Blond, or Red)\n")
color_pref = color_pref.lower()

#MAIN CODE
listener.start()

if __name__ == '__main__':
    del_pics() # Deletes all pictures incase something went wrong
    while listener.is_alive():
        screen_face_detect.screenshot() # Takes a screenshot of the desired size

        screen_face_detect.detect_faces(screenshot_path) # Detects the images in the screenshot

        if face_counter() > 1 or face_counter()==0:
            continue # Begin the loop again if more than 1 person in the picture

        model_predict.extract_features(face_path) # ML algo extracts features for analysis

        swipe(model_predict.predict_hair_color(face_path)) # Pre-trained ML algo predicts hair color
        del_pics() # Deletes all pics  
listener.stop()
