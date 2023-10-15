"""
This script changes the PNG images provided below to the same folder structure as the original data set.  
Get PNG Images from: https://www.kaggle.com/code/theoviel/get-started-quicker-dicom-png-conversion/notebook

**Dataset Links**
- Part 1 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt1
- Part 2 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt2
- Part 3 : https://www.kaggle.com/datasets/theoviel/rsna-2023-abdominal-trauma-detection-pngs-3-8
- Part 4 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt4
- Part 5 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt5
- Part 6 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-png-pt6
- Part 7 : https://www.kaggle.com/datasets/theoviel/rsna-abdominal-trauma-detection-pngs-pt7
- Part 8 : https://www.kaggle.com/datasets/theoviel/rsna-2023-abdominal-trauma-detection-pngs-18

"""

import shutil
from pathlib import Path
from tqdm import tqdm

# the folder where unzipped the zipped image file
data_dir = Path(r"D:\RSNA2023\data")
train_images_dir = data_dir / "train_images"
train_images_path = [
    image_path for image_path in train_images_dir.glob("*.png")
]

for image_path in tqdm(train_images_path):
    patient_id, series_id, image_id = image_path.stem.split("_")
    train_patient_series_dir = (train_images_dir / patient_id / series_id)

    if not (train_patient_series_dir.is_dir()):
        train_patient_series_dir.mkdir(parents=True)

    shutil.move(image_path, train_patient_series_dir)
