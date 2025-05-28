import os
import cv2 as cv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle


root_path = Path("/Volumes/Big Data/data/image_data")
MAX_BATCH_FILE_SIZE_GB = 2
IMG_DIM = (32, 32)
MAX_BATCH_SIZE = int((MAX_BATCH_FILE_SIZE_GB*1024**3)/(IMG_DIM[0]*IMG_DIM[1]*4)) #4 bytes for float32

batch_data = np.empty((MAX_BATCH_SIZE, 1, 32, 32), dtype=np.float32)
img_index = 0
batch_number = 0

#root_path = Path("images")
output_dir = Path("ds_32_32_INTER_AREA")
output_dir.mkdir(parents=True, exist_ok=True)


def process_img(img_path: Path) -> None:
    try:
        img = cv.imread(str(img_path),cv.IMREAD_GRAYSCALE)
        if img is None:
            return
        img = np.float32(img)/255
        img_resized = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
        img_resized = np.clip(img_resized, 0, 1)
        rez = np.reshape(img_resized, (1, 32, 32))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
    return rez


def save_batch(arr, count, idx):
    save_path = output_dir / f"batch_{idx:04d}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(arr[:count], f)
    print(f"Saved {count} images to {save_path}")


image_paths = (
    Path(root) / file
    for root, _, files in os.walk(root_path)
    for file in files
    if file.lower().endswith(('.jpg', '.png', '.jpeg'))
)

with ThreadPoolExecutor(max_workers=8) as executor:
    try:
        for img in executor.map(process_img, image_paths):
            if img is None:
                continue
            batch_data[img_index] = img
            img_index += 1
            
            if img_index == MAX_BATCH_SIZE:
                save_batch(batch_data, img_index, batch_number)
                img_index = 0
                batch_number += 1
            if img_index % 1000 == 0:
                print(img_index)
    except KeyboardInterrupt:
        save_batch(batch_data, img_index, batch_number)
            
if img_index > 0:
    save_batch(batch_data, img_index, batch_number)