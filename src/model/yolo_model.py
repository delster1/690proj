from ultralytics import YOLO
import tensorflow as tf

class YoloModel:

    def __init__(self, model_type="yolov8n.pt"):
        """
        model_type: 'yolov8n.pt', 'yolov8s.pt', 'yolov8n.yaml' (from scratch)
        """
        self.model = YOLO(model_type)

    def train(self, data="dataset/data.yaml", epochs=50):
        self.model.train(
            data=data,
            epochs=epochs,
            imgsz=640,             # recommended
            batch=8,
            device=0 if tf.config.list_physical_devices('GPU') else 'cpu'
        )

    def export_tflite(self):
        print("✔ Exporting to TFLite INT8...")
        self.model.export(format="tflite", int8=True)
        print("✔ Export complete!")

    def infer(self, image_path):
        return self.model(image_path)

