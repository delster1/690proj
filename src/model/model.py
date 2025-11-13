import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_CLASSES = 10


class CNN:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
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
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

    def train(self, train_dir, val_dir):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
             rotation_range=40,
             width_shift_range=0.2,
             height_shift_range=0.2,
             shear_range=0.2,
             zoom_range=0.2,
             horizontal_flip=True,
             fill_mode='nearest'
         )

        train_generator=train_datagen.flow_from_directory(
            # This is the source directory for training images
            train_dir,
            target_size=(1920, 1080),  # All images will be resized to 100x100
            batch_size=128,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical_crossentropy')


        # VALIDATION
        validation_datagen=ImageDataGenerator(rescale=1./255)

        validation_generator=validation_datagen.flow_from_directory(
            val_dir,
            target_size=(1920,1080),
            class_mode='categorical_crossentropy'
        )
