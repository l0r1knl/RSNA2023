from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pydicom

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

import torchvision
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms
torchvision.disable_beta_transforms_warning()


class AbdominalTraumaDataset(Dataset):
    def __init__(
        self,
        patient_info: pd.DataFrame,
        patient_series: pd.DataFrame,
        img_root_dir: Path,
        img_extension: str = ".dcm",
        transform: Optional[transforms.Transform] = None,
        pre_transform: Optional[transforms.Transform] = None,
        target_transform: Optional[transforms.Transform] = None,
        is_pseudo3D: bool = True,
        has_pseudo3D_img: bool = False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:

        self.patient_info = patient_info
        self.patient_series = patient_series
        self.img_root_dir = img_root_dir
        self.img_extension = img_extension
        self.device = device
        self.is_train = "any_injury" in self.patient_info.columns
        self.is_pseudo3D = is_pseudo3D
        self.has_pseudo3D_img = has_pseudo3D_img

        self.transform = transform.to(device) if transform else transform
        self.pre_transform = pre_transform.to(
            device) if pre_transform else pre_transform
        self.target_transform = target_transform.to(
            device) if target_transform else target_transform

        if self.is_train:
            self.__set_multi_labels()
        else:
            self.labels = torch.empty(self.patient_series.shape)
            self.classes = None

    def __get_images_dir(
        self,
        patient_id: int,
        series_id: int,
    ) -> Path:
        return (self.img_root_dir / str(patient_id) / str(series_id))

    def __read_images(
        self,
        images_dir: Path,
    ) -> list[torch.Tensor]:
        return [
            read_image(str(path)).to(self.device)
            for path in sorted(images_dir.glob(f"*{self.img_extension}"))
        ]

    def __read_images_from_dicom(
        self,
        images_dir: Path,
        size: Optional[int] = 512
    ) -> list[torch.Tensor]:

        imgs = {}
        for path in sorted(images_dir.glob(f"*{self.img_extension}")):
            dicom = pydicom.dcmread(path)

            pos_z = dicom[(0x20, 0x32)].value[-1]

            img = standardize_pixel_array(dicom)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = 1 - img

            img = (img * 255).astype(np.uint8)
            imgs[pos_z] = cv2.resize(
                img, (size, size)) if size is not None else img

        return [torch.from_numpy(imgs[k]).to(self.device) for k in sorted(imgs.keys())]

    def __read_pseudo3D_image(
        self,
        images_dir: Path,
        patient_id: int,
        series_id: int
    ) -> torch.Tensor:
        image_path = (
            images_dir / str(patient_id) /
            (str(series_id) + self.img_extension)
        )
        return read_image(str(image_path)).to(self.device)

    def __set_multi_labels(self) -> None:
        labels = pd.merge(
            self.patient_series,
            self.patient_info,
        )

        labels["kidney_injury"] = (
            labels["kidney_low"] + labels["kidney_high"] * 2
        )
        labels["liver_injury"] = labels["liver_low"] + labels["liver_high"] * 2
        labels["spleen_injury"] = (
            labels["spleen_low"] + labels["spleen_high"] * 2
        )

        self.labels = torch.Tensor(
            labels.loc[
                :,
                [
                    "bowel_injury",
                    "extravasation_injury",
                    "kidney_injury",
                    "liver_injury",
                    "spleen_injury",
                    "any_injury",
                    "incomplete_organ"
                ]
            ].copy().values
        )

        self.classes = labels.columns

    def __getitem__(self, idx):
        patient_id = self.patient_series.iloc[idx, 0]
        series_id = self.patient_series.iloc[idx, 1]
        images_dir = self.__get_images_dir(patient_id, series_id)

        if self.has_pseudo3D_img and self.is_pseudo3D:
            image = self.__read_pseudo3D_image(
                self.img_root_dir, patient_id, series_id
            )
        else:
            if self.img_extension == ".dcm":
                image = self.__read_images_from_dicom(images_dir)
            else:
                image = self.__read_images(images_dir)

            if self.is_pseudo3D:
                image = make_pseudo3D_CT_image(image, self.device)

        label = self.labels[idx, :].to(self.device)

        if self.transform:
            image = self.transform(image)

        if self.target_transform and label is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return self.patient_series.shape[0]


def standardize_pixel_array(
    dcm: pydicom.dataset.FileDataset
) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2

    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array


def make_pseudo3D_CT_image(
    images: list[torch.Tensor],
    device: torch.device
) -> torch.Tensor:

    for i, image in enumerate(images):
        if i == 0:
            pseudo3D_image = (image.detach().clone() / 255.0).to(device)
        else:
            pseudo3D_image += (image / 255.0)
    pseudo3D_image = (pseudo3D_image / len(images) * 255).to(torch.uint8)

    return pseudo3D_image.repeat(3, 1, 1)


def make_datalodaers(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
) -> dict[DataLoader]:

    return dict(
        train=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ),
        valid=DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    )


def delete_series(
        patient_series: pd.DataFrame,
        image_dir: Path
) -> pd.DataFrame:
    return patient_series.iloc[
        [row.Index for row in patient_series.itertuples() if (image_dir / str(row.patient_id) / str(row.series_id)).exists()], :
    ].reset_index(drop=True)
