from yolo_model import YoloModel

def main():
    model = YoloModel("yolov8n.pt")

    print(">>> Training model...")
    model.train(epochs=50)

    print(">>> Exporting shrunk INT8 model...")
    model.export_tflite()

    print(">>> Testing inference...")
    results = model.infer("dataset/images/val/some_image.jpg")
    print(results)

if __name__ == "__main__":
    main()

