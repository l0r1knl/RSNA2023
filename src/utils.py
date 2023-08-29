import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import F


def plot_images(images: torch.Tensor):
    if images.ndim == 3:
        images = [images]

    fig, axes = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axes[0, i].imshow(np.asarray(img))
        axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.close()

    return fig
