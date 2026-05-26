import numpy as np
import json
from tensorflow import tf
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input


with open("disease_db.json", "r") as f:
    disease_db = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

interpreter = tf.lite.Interpreter(model_path="plant_disease.tflite")
interpreter.allocate_tensors()


def preprocess_image(image_path):
    image = Image.open(image_path)
    fixed_image = image.resize((224, 224))
    image_array = preprocess_input(np.array(fixed_image))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict(image_path):
    image_array = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    confidence = float(np.max(output))
    disease_name = disease_db[str(np.argmax(output))]
    return (disease_name,
            confidence,
            disease_info[disease_name]["treatment"],
            disease_info[disease_name]["description"],
            disease_info[disease_name]["prevention"]
            )
