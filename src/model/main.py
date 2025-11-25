from yolo_model import YoloModel

def main():
    model = YoloModel(classes=10)
    model.set_data_generators(train_gen, val_gen)
    model.train(epochs=50)
    model.shrink_model()

if __name__ == "__main__":
    main()
