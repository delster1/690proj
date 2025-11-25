import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_CLASSES = 3


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
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            # This is the source directory for training images
            train_dir,
            target_size=(1920, 1080),  # All images will be resized to 100x100
            batch_size=128,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical_crossentropy')

        # VALIDATION
        self.validation_datagen = ImageDataGenerator(rescale=1./255)

        self.validation_generator = self.validation_datagen.flow_from_directory(
            val_dir,
            target_size=(1920, 1080),
            class_mode='categorical_crossentropy'
        )

    def save_model(self):
        self.model.save("saved_model")   # creates folder ./saved_model

    def shrink_model(self):
        self.save_model()

        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set representative dataset
        converter.representative_dataset = lambda: self.representative_data_gen()

        # Force full INT8 quantization for inference on embedded devices
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Force model input/output to be int8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        tflite_model_file = 'converted_model.tflite'

        with open(tflite_model_file, "wb") as f:
            f.write(tflite_model)

    def representative_data_gen(self):
        for i in range(100):  # take ~100 batches for calibration
            image_batch, _ = next(self.validation_generator)

            # TFLite expects a list of input tensors
            # Use only 1 sample per iteration for stability
            yield [image_batch[0:1]]
