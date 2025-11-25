from yolov4.tf import YOLOv4
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_DIR = "animals/images/animals-10"
IMG_SIZE = (1920, 1080)


class YoloModel:

    def __init__(self, input_size=(1920,1080), classes=10, tiny=False):
        self.input_size = input_size
        self.classes = classes
        
        self.tiny = tiny
        
        self.yolo = YOLOv4(tiny=self.tiny)
        self.yolo.classes = self.classes
        self.yolo.input_size = self.input_size

        self.yolo.make_model()

        if (self.tiny == False):
            self.yolo.load_weights("yolov4.weights", weights_type="yolo")
        else:
            self.yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")

        self.yolo.compile(
            loss="yolo_loss",
            optimizer=tf.keras.optimizers.Adam(1e-4)
        )

        self.train_gen = None
        self.val_gen = None

    def set_data_generators(self, train_gen, val_gen):
        self.train_gen = train_gen
        self.val_gen = val_gen


    def train_from_saved(self, epochs=50):

        # 1. Rebuild YOLOv4 architecture
        yolo = YOLOv4(tiny=self.tiny)
        yolo.classes = self.classes
        yolo.input_size = self.input_size
        yolo.make_model()

        # 2. Load saved weights
        if self.tiny:
            yolo.load_weights("tiny_fine_tuned.h5")
        else:
            yolo.load_weights("fine_tuned.h5")

        # 3. Compile with YOLO loss + optimizer
        yolo.compile(
            loss="yolo_loss",
            optimizer=tf.keras.optimizers.Adam(1e-4)
        )

        # 4. Train
        yolo.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs
        )

        # 5. Save updated weights
        retrained_path = ("tiny_" if self.tiny else "") , ("retrained_model.h5")
        yolo.save_weights("retrained_model.h5")

        return yolo

    def train(self, epochs=50):
            if self.train_gen is None or self.val_gen is None:
                raise ValueError("Generators must be set first.")
            
            self.yolo.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=epochs
            )

            if self.tiny == False:
                self.yolo.save_weights("fine_tuned.h5")
            else:
                self.yolo.save_weights("tiny_fine_tuned.h5")

    def export_saved_model(self, path="saved_model"):
        if not self.tiny:
            self.yolo.save(path)
        else:
            self.yolo.save("tiny_saved_model")

    def shrink_model(self):
        saved_model_name = "saved_model" if not self.tiny else "tiny_saved_model"
        self.export_saved_model(saved_model_name)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_name)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset (your partner's generator)
        converter.representative_dataset = self.representative_data_gen

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        model_destination = "tiny_" if self.tiny else ""
        model_path = f"{model_destination}yolov4_int8.tflite"
        with open(model_path, "wb") as f:
            f.write(tflite_model)

        print(f"Model successfully quantized → {model_path}")
        self.mini_model = tf.lite.Interpreter(model_path=model_path)
        self.mini_model.allocate_tensors()

    def representative_data_gen(self):
        """
        Use the validation generator to provide representative images
        for post-training quantization.
        """
        for i in range(100):
            imgs, _ = next(iter(self.val_gen))
            # Single sample each iteration
            yield [imgs[:1].astype("float32")]

    def run_mini_inference(self, image):
        input_details = self.mini_model.get_input_details()
        output_details = self.mini_model.get_output_details()

        self.mini_model.set_tensor(input_details[0]['index'], image)
        self.mini_model.invoke()
        output = self.mini_model.get_tensor(output_details[0]['index'])
        return output

