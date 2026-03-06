
import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = "./animals"
OUTPUT_DIR = "./dataset"
TRAIN_SPLIT = 0.8

def main():
    # Clean old dataset
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    images_train = Path(f"{OUTPUT_DIR}/images/train")
    images_val   = Path(f"{OUTPUT_DIR}/images/val")
    labels_train = Path(f"{OUTPUT_DIR}/labels/train")
    labels_val   = Path(f"{OUTPUT_DIR}/labels/val")

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Load all label pairs
    samples = []

    for class_folder in os.listdir(SOURCE_DIR):
        folder = f"{SOURCE_DIR}/{class_folder}"
        label_folder = f"{folder}/Label"

        if not os.path.isdir(folder) or not os.path.isdir(label_folder):
            continue

        for f in os.listdir(folder):
            if f.endswith((".jpg", ".jpeg", ".png")):
                image_path = f"{folder}/{f}"
                label_path = f"{label_folder}/{f.rsplit('.',1)[0]}.txt"
                samples.append((image_path, label_path))

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)

    train = samples[:split_idx]
    val = samples[split_idx:]

    # Copy into YOLOv8 structure
    for is_train, subset in [(train, "train"), (val, "val")]:
        for img_path, label_path in is_train:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(label_path)

            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{subset}/{img_name}")
            shutil.copy(label_path, f"{OUTPUT_DIR}/labels/{subset}/{lbl_name}")

    print("✔ Dataset successfully converted for YOLOv8.")

if __name__ == "__main__":
    main()

