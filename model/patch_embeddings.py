""" Contains functions and classes for generating input for Transformer Encoder"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary


def generate_patches(imgs: torch.Tensor, patch_size: int) -> torch.tensor:
    """Generate patches from an image. Image size must be channel, height, width. Height and width must be equal and be divisible by patch_size.

    Args:
        img (torch.Tensor): Image to generate patches from.
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Patch of image. dim - channel, patch_size, patch_size.
    """

    # Check if image is batch, channel, height, width
    assert imgs.ndimension() == 3, "Image must be channel, height, width"

    # Check if height and width are equal and divisible by patch_size
    assert (
        imgs.size(1) == imgs.size(2) and imgs.size(2) % patch_size == 0
    ), "Height and width must be equal and divisible by patch_size"

    # Generate patches
    patches = (
        imgs.unfold(0, imgs.size(0), imgs.size(0))  # Unfold channel dimension
        .unfold(1, size=patch_size, step=patch_size)  # Unfold height dimension
        .unfold(2, size=patch_size, step=patch_size)  # Unfold width dimension
    )

    # Reshaping to batch, patch, channel, patch_size, patch_size
    patches = patches.reshape(
        (
            (imgs.size(2) // patch_size) ** 2,
            imgs.size(0),
            patch_size,
            patch_size,
        )
    )

    return patches


class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        patch_size: int,
        image_size: int,
        in_channels: int = 3,
        embedding_dim: int = 768,
    ):
        """Generating theinput for Transformer Encoder. Image size must be channel, height, width. Height and width must be equal and be divisible by patch_size.

        1. The input first is passed through the convolutional layer and then the embeddings are generated and reshaped into `batch, no of patches, embedding_dim`.
        2. The embeddings are then concatenated with the class embedding, dim - `batch, no of patches+1, embedding_dim`.
        3. positional embeddings and added to the embeddings, dim - `batch, no of patches+1, embedding_dim`.

        Args:
            patch_size (int): Size of each patch.
            image_size (int): Size of the input image.
            in_channels (int, optional): Number of channels in the input image. Defaults to 3.
            embedding_dim (int, optional): Dimension of the embedding. Defaults to 768.
        """
        super().__init__()

        self.patch_embeddings = nn.Sequential(
            # Convolutional layer
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embedding_dim,  # Converting the image into a vector of embeddings
                kernel_size=patch_size,  # Size of the kernel is the size of the patch,
                stride=patch_size,  # Stride is the same as the patch size
                padding=0,
            ),
            nn.Flatten(
                2, 3
            ),  # Flatten the output of the convolutional layer into a 1d vector
        )

        # Adding an extra learnable class embedding as mentioned in the paper
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embedding_dim),  # batch size, 1, embedding_dim
            requires_grad=True,  # the class embedding parameters will update through backpropagation
        )

        # Adding positional embeddings in class embedding and each patch
        self.positional_embeddings = nn.Parameter(
            torch.rand(
                1, ((image_size // patch_size) ** 2) + 1, embedding_dim
            ),  # batch size, (number of patches + 1) , embedding_dim.
            # +1 because of addition class token dim in the embedding
            requires_grad=True,  # the positional embedding are learnable and will update through backpropagation
        )

    def forward(self, img: torch.tensor) -> torch.tensor:
        """Forward pass of the model.

        1. The input first is passed through the convolutional layer and then the embeddings are generated and reshaped into `batch, no of patches, embedding_dim`.
        2. The embeddings are then concatenated with the class embedding, dim - `batch, no of patches+1, embedding_dim`.
        3. positional embeddings and added to the embeddings, dim - `batch, no of patches+1, embedding_dim`.

        Args:
            img (torch.tensor): Image to generate patches from. dim - batch, channel, height, width.

        Returns:
            torch.tensor: Patch embeddings. dim - batch, number of patches, embedding_dim."""

        batch_size = img.size(0)

        x = self.patch_embeddings(img)
        x = x.permute(0, 2, 1)

        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # adding more batch size to the class embedding
        x = torch.cat((cls_token, x), dim=1)

        positional_embeddings = self.positional_embeddings.expand(
            batch_size, -1, -1
        )  # adding more batch size to the positional embedding
        x = x + positional_embeddings

        return x


if __name__ == "__main__":

    PATCH_SIZE = 14
    IMAGE_SIZE = 28

    # Reading the image
    img = cv2.imread("utils/test_imgs/saturnv.jpg")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    # Splitting the image into patches for visualization
    img = torch.tensor(img.astype(np.float32) / 255.0).transpose(-1, 0)
    patches = generate_patches(img, PATCH_SIZE)
    out_img = torchvision.utils.make_grid(
        patches, nrow=img.size(2) // PATCH_SIZE, padding=10
    )
    # plt.imshow(out_img.transpose(0, -1))
    # plt.show()

    # Generating the input for Transformer Encoder
    patch_embed = PatchEmbeddings(
        PATCH_SIZE,
        image_size=IMAGE_SIZE,
        in_channels=3,
        embedding_dim=768,
    )

    summary(
        patch_embed,
        input_data=(3, IMAGE_SIZE, IMAGE_SIZE),
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
        depth=2,
    )
