import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications.mobilenet_v2 import preprocess_input


if __name__ == "__main__":
    data = image_dataset_from_directory(
        directory=r"C:\Users\moshe\Documents\Projects\Plant Desease Detection\data\raw\plantvillage dataset\color",
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="both",
        seed=100,
        )
    train_ds, val_ds = data
    half = len(val_ds) // 2
    test_ds = val_ds.skip(half)
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)
    model = tf.keras.models.load_model("best_model.keras")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
