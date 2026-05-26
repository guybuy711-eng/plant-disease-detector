import tkinter as tk
from PIL import Image, ImageTk
from camera import capture_and_predict


def show_result(prediction, confidence, treatment, description, prevention, image_path):
    window = tk.Tk()
    window.title("Plant Disease Detector")

    img = Image.open(image_path)
    img = img.resize((300, 300))
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(window, image=photo)
    label.pack()

    tk.Label(window, text="Prediction On Suspected Disease:").pack()
    prediction_label = tk.Label(window, text=prediction)
    prediction_label.pack()

    tk.Label(window, text="Confidence On Suspected Disease:").pack()
    confidence_label = tk.Label(window, text=f"{confidence * 100:.1f}%")
    confidence_label.pack()

    tk.Label(window, text="Details On Treatment:").pack()
    for detail in treatment:
        treatment_label = tk.Label(window, text=detail)
        treatment_label.pack()

    tk.Label(window, text="Description On Disease:").pack()
    description_label = tk.Label(window, text=description)
    description_label.pack()

    tk.Label(window, text="How To Prevent:").pack()
    prevention_label = tk.Label(window, text=prevention)
    prevention_label.pack()

    window.mainloop()

if __name__ == "__main__":
    prediction, confidence, treatment, description, prevention = capture_and_predict()
    show_result(prediction, confidence, treatment, description, prevention, "temp.jpg")
