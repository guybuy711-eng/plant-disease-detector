import tensorflow as tf
import json
from keras.utils import image_dataset_from_directory
from keras.applications.mobilenet_v2 import preprocess_input
from models.model import build_model


if __name__ == "__main__":
    model = build_model()

    data = image_dataset_from_directory(
        directory=r"C:\Users\moshe\Documents\Projects\Plant Desease Detection\data\raw\plantvillage dataset\color",
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="both",
        seed=100,
    )

    train_ds, val_ds = data

    disease_db = dict(enumerate(train_ds.class_names))

    with open("disease_db.json", "w") as f:
        json.dump(disease_db, f)

    half = len(val_ds) // 2
    test_ds = val_ds.skip(half)
    val_ds = val_ds.take(half)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callback = [
        tf.keras.callbacks.ModelCheckpoint(filepath="best_model.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callback,
        epochs=5
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("plant_disease.tflite", "wb") as f:
        f.write(tflite_model)
