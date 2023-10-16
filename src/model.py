from pathlib import Path
from typing import (
    OrderedDict,
    Optional
)
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18
torchvision.disable_beta_transforms_warning()


class MultiAbdominalTraumaClassifier(nn.Module):
    """ My baseline model for RSNA2023.
    """

    def __init__(
        self,
        n_bowel_status: int = 2,
        n_extravasation_status: int = 2,
        n_kidney_status: int = 3,
        n_liver_status: int = 3,
        n_spleen_status: int = 3,
        n_anyinjury_status: int = 2,
        n_incomplete: int = 2,
        backbone: nn.Module = resnet18(weights="IMAGENET1K_V1"),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        backbone_lastlayer = list(backbone.children())[-1]
        if type(backbone_lastlayer) is torch.nn.modules.container.Sequential:
            backbone_lastlayer = list(backbone_lastlayer.children())[-1]

        num_ftrs = backbone_lastlayer.in_features

        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))

        self.is_transformer = backbone._get_name() == "VisionTransformer"
        if self.is_transformer:
            self._process_input = backbone._process_input
            self.class_token = backbone.class_token
        else:
            self.flatten = nn.Flatten()

        self.bowel_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_bowel_status)
        )
        self.extravasation_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs,
                      out_features=n_extravasation_status)
        )
        self.kidney_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_kidney_status)
        )
        self.liver_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_liver_status)
        )
        self.spleen_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_spleen_status)
        )
        self.any_injury_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_anyinjury_status)
        )
        self.incomplete_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_ftrs, out_features=n_incomplete)
        )

    def forward(self, x):
        if self.is_transformer:
            x = self._transformer_process(x)
        else:
            x = self.backbone(x)
            x = self.flatten(x)

        return {
            "bowel_injury": self.bowel_classifier(x),
            "extravasation_injury": self.extravasation_classifier(x),
            "kidney_injury": self.kidney_classifier(x),
            "liver_injury": self.liver_classifier(x),
            "spleen_injury": self.spleen_classifier(x),
            "any_injury": self.any_injury_classifier(x),
            "incomplete_organ": self.incomplete_classifier(x),
        }

    def _transformer_process(self, x):

        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone[1](x)
        x = x[:, 0]

        return x

    def fit(
        self,
        dataloaders: dict[str:DataLoader],
        criterions: dict[_Loss],
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        num_epochs: int,
        criterions_weight: Optional[dict[float]] = None,
        save_dir: Optional[Path] = None,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    ):
        best_loss = 10000
        epoch_losses = dict()
        self.to(device)

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.train()  # Set model to training mode
                else:
                    self.eval()   # Set model to evaluate mode

                running_loss = None
                total = len(dataloaders[phase])

                # Iterate over data.
                with tqdm(total=total) as pbar:
                    pbar.set_description(
                        f'[Epoch: {epoch + 1}/{num_epochs}, Phase: {phase}]'
                    )

                    for inputs, labels in dataloaders[phase]:
                        pbar.update(1)
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self(inputs)
                            losses = self.calc_losses(
                                criterions, outputs, labels, criterions_weight
                            )
                            # backward + optimize only if in training phase
                            if phase == "train":
                                losses["average_loss"].backward()
                                optimizer.step()

                        if running_loss is None:
                            running_loss = {
                                label: loss.item() * inputs.size(0)
                                for label, loss in losses.items()
                            }
                        else:
                            for label, loss in losses.items():
                                running_loss[label] += (
                                    loss.item() * inputs.size(0)
                                )
                        pbar.set_postfix(
                            {'loss (in progress)': running_loss["average_loss"]}
                        )

                    dataset_size = len(dataloaders[phase].dataset)
                    _epoch_loss = {
                        f"{phase}_{label}": loss / dataset_size
                        for label, loss in running_loss.items()
                    }

                    if phase == "train":
                        epoch_losses[epoch] = _epoch_loss
                    else:
                        epoch_losses[epoch].update(_epoch_loss)

                    if scheduler and phase == "train":
                        scheduler.step(epoch)

                    if phase == "valid" and epoch_losses[epoch][f"{phase}_average_loss"] <= best_loss:
                        best_loss = epoch_losses[epoch][f"{phase}_average_loss"]
                        best_epoch = epoch

                    if save_dir:
                        self.save_model_state(
                            save_dir / f"E{epoch:03}.pt", device
                        )
                    pbar.set_postfix(
                        {'loss (final)': epoch_losses[epoch][f"{phase}_average_loss"]}
                    )
                    pbar.close()

        if save_dir:
            self.load_model_state(save_dir / f"E{best_epoch:03}.pt", device)

        return epoch_losses

    def calc_losses(
        self,
        criterions: dict[_Loss],
        outputs: list[torch.Tensor],
        labels: list[torch.Tensor],
        criterions_weight: Optional[dict[float]] = None,
    ) -> dict[str:torch.Tensor]:

        loss = 0.0
        losses = dict()

        for i, key in enumerate(outputs):
            weight = (criterions_weight[key] if criterions_weight else 1)
            losses[key] = criterions[key](
                outputs[key].softmax(dim=1), labels[:, i].long())
            loss += losses[key] * weight

        loss /= len(criterions)

        losses.update({f"average_loss": loss})

        return losses

    def save_model_state(self, path: Path, device: torch.device) -> None:
        self.to(torch.device("cpu"))
        torch.save(self.state_dict(), path)
        self.to(device)

    def load_model_state(self, path: Path, device: torch.device) -> None:
        self.to(torch.device("cpu"))
        self.load_state_dict(torch.load(path))
        self.to(device)

    def predict(self, data: Dataset) -> list[np.ndarray]:
        self.eval()
        loader = DataLoader(data, batch_size=1, shuffle=False)

        proba = []
        for inputs, _ in tqdm(loader):
            outputs = self(inputs)
            for i, (label, prob) in enumerate(outputs.items()):
                if i == 0:
                    output = prob.softmax(dim=1).detach().clone()
                else:
                    output = torch.hstack(
                        (output, prob.softmax(dim=1).detach().clone()))
            proba.append(output.to("cpu").numpy()[0])

        return proba
