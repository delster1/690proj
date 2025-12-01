import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random


class YoloDataGenerator(Sequence):

    def __init__(
        self,
        dataset_dir,          # e.g. "animals/"
        input_size=(416, 416),
        batch_size=8,
        max_boxes=50,
        shuffle=True,
        augment=False,
        val_split=0.2,
        subset="train"
    ):
        self.dataset_dir = dataset_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.shuffle = shuffle
        self.augment = augment
        self.subset = subset
        self.val_split = val_split

        self.samples = []

        for class_folder in os.listdir(dataset_dir):
            full_path = os.path.join(dataset_dir, class_folder)

            # skip non-directories
            if not os.path.isdir(full_path):
                continue

            label_folder = os.path.join(full_path, "labels")
            if not os.path.isdir(label_folder):
                continue

            # collect all .jpg/.png images in this class folder
            for f in os.listdir(full_path):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(full_path, f)
                    label_path = os.path.join(label_folder, f.rsplit('.', 1)[0] + ".txt")
                    self.samples.append((img_path, label_path))

        # sort for consistency
        self.samples = sorted(self.samples)

        total = len(self.samples)
        split = int(total * (1 - val_split))

        if subset == "train":
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]

        # create index array
        self.indexes = np.arange(len(self.samples))
        self.on_epoch_end()


    def __len__(self):
        return len(self.indexes) // self.batch_size


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __getitem__(self, idx):
        batch_idxs = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = []
        boxes  = []

        for j in batch_idxs:
            img_path, label_path = self.samples[j]

            img, b = self.load_image_and_labels(img_path, label_path)

            if self.augment:
                img, b = self.apply_augmentations(img, b)

            images.append(img)
            boxes.append(self.pad_boxes(b))

        return np.array(images, dtype="float32"), np.array(boxes, dtype="float32")


    def load_image_and_labels(self, img_path, label_path):

        # -------------------------------------------------
        # Load original image (PIL)
        # -------------------------------------------------
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # -------------------------------------------------
        # LETTERBOX RESIZE
        # -------------------------------------------------
        target_w, target_h = self.input_size  # e.g. (416, 416)
        scale = min(target_w / orig_w, target_h / orig_h)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # resize while preserving aspect ratio
        resized = img.resize((new_w, new_h), Image.BICUBIC)

        # create padded canvas
        # YOLO pads equally on all sides
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2

        padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded.paste(resized, (dx, dy))

        # convert to numpy
        img = np.array(padded, dtype=np.float32) / 255.0

        # -------------------------------------------------
        # Load YOLO TXT labels
        # -------------------------------------------------
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.split())

                    # ------------------------------------------
                    # Un-normalize to original image pixels
                    # ------------------------------------------
                    cx *= orig_w
                    cy *= orig_h
                    w  *= orig_w
                    h  *= orig_h

                    # ------------------------------------------
                    # Apply letterbox scaling
                    # ------------------------------------------
                    cx = cx * scale + dx
                    cy = cy * scale + dy
                    w  = w  * scale
                    h  = h  * scale

                    # ------------------------------------------
                    # Re-normalize to squared letterboxed dimensions
                    # ------------------------------------------
                    cx /= target_w
                    cy /= target_h
                    w  /= target_w
                    h  /= target_h

                    boxes.append([cls, cx, cy, w, h])

        return img, np.array(boxes, dtype=np.float32)



    def apply_augmentations(self, img, boxes):
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            if len(boxes) > 0:
                boxes[:, 1] = 1 - boxes[:, 1]  # flip cx
        return img, boxes


    def pad_boxes(self, boxes):
        padded = np.zeros((self.max_boxes, 5))
        if len(boxes) > 0:
            padded[: len(boxes)] = boxes[: self.max_boxes]
        return padded

