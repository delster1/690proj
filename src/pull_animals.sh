#!/bin/bash

set -e 

TARGET_DIR=animals/images/animals-detection-images
DATASET_NAME=antoreepjana/animals-detection-images-dataset
ZIP_NAME="animals10.zip"

if ! command -v kaggle &> /dev/null; then
  echo "[ERROR] Kaggle cli not installed"
  echo "Install with: pip install kaggle"
fi

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"


echo "[INFO] Downloading dataset: $DATASET"

if [ -f "$ZIP_NAME" ]; then
  echo "[INFO] $ZIP_NAME already exists! Skipping download..."
else
  kaggle datasets download -d "$DATASET_NAME" -p . --force
fi

echo "[INFO] Unzipping dataset"
unzip -o "$ZIP_NAME"

if [ ! -d "raw-img" ]; then
    echo "[ERROR] 'raw-img' folder not found after unzip."
    exit 1
fi

echo "[INFO] Structuring dataset"

mv raw-img/* .
rmdir raw-img

echo "[INFO] Cleaning up zip file..."
rm -f "$ZIP_NAME"

echo "[DONE] animals10 dataset ready in $TARGET_DIR"



