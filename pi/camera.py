from predict import predict
from picamera2 import Picamera2
import time


def capture_and_predict():
    camera = Picamera2()
    camera.start()

    while True:
        camera.capture_file("temp.jpg")
        prediction, confidence, treatment, description, prevention = predict("temp.jpg")
        if confidence > 0.8:
            return prediction, confidence, treatment, description, prevention
        time.sleep(1)
