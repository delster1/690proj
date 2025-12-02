from ultralytics import YOLO
import tensorflow as tf
import albumentations as A

class YoloModel:

    def __init__(self, model_type="yolov8n.pt"):
        """
        model_type: 'yolov8n.pt', 'yolov8s.pt', 'yolov8n.yaml' (from scratch)
        """
        self.model = YOLO(model_type)

    def train(self, data="dataset/data.yaml", epochs=50):
        seimport albumentations as A

        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.GaussNoise(p=0.3),
            A.Blur(p=0.2),
            A.Cutout(num_holes=2, max_h_size=32, max_w_size=32, p=0.2),
        ])

        lf.model.train(
            data=data,
            epochs=epochs,
            imgsz=640,           
            batch=8,
            qat=True,
            transform=transforms
            device=0 if tf.config.list_physical_devices('GPU') else 'cpu'
        )

    def export_tflite(self):
        print("✔ Exporting to TFLite INT8...")
        self.model.export(format="tflite", int8=True)
        print("✔ Export complete!")

    def infer(self, image_path):
        return self.model(image_path)

