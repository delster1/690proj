import tensorflow as tf


class CNN:
    def __init__(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', input_shape=(1920, 1080, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
