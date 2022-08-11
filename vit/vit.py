""" Implementing the Vision Transformer with combining encoder blocks and MLP head
"""

# Importing Libraries
import torch
import torch.nn as nn
from torchsummary import summary

from vit.patch_embeddings import PatchEmbeddings
from vit.encoder import Encoder
from vit.mlp_head import MLPHead


class ViT(nn.Module):
    def __init__(
        self,
        patch_size: int,
        image_size: int,
        in_channels: int,
        n_classes: int,
        embed_dim: int,
        n_encoder: int,
        n_heads: int,
        mlp_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
    ):
        """Implementation of the Vision Transformer.

        Args:
            patch_size (int): Size of the input patch.
            image_size (int): Size of the input image.
            in_channels (int): Number of channels in the input image.
            n_classes (int): Number of classes in the output.
            embed_dim (int): Dimension of the embedding.
            n_encoder (int): Number of encoder blocks.
            n_heads (int): Number of attention heads.
            mlp_dropout_p (float, optional): Dropout probability for the MLP block. Defaults to 0.1.
            attn_dropout_p (float, optional): Dropout probability for the attention block. Defaults to 0.
        """

        super().__init__()

        self.patch_embeddings = PatchEmbeddings(
            patch_size=patch_size,
            image_size=image_size,
            in_channels=in_channels,
            embedding_dim=embed_dim,
        )
        self.encoders = nn.ModuleList(
            [
                Encoder(embed_dim, n_heads, mlp_dropout_p, attn_dropout_p)
                for _ in range(n_encoder)
            ]
        )

        self.mlp_head = MLPHead(embed_dim, n_classes)

    def forward(self, x):
        """Computes the feed forward for the Vision Transformer.

        Steps -
        1. Get the embedding for the input patch.
        2. Pass the embedding through the encoder blocks.
        3. Pass the embedding through the MLP head, using 0th index of the last encoder output.
        4. Output the logits.

        Args:
            x: Tensor of shape [batch_size, channels, height, width]

        Returns:
            logits: Tensor of shape [batch_size, n_classes]
        """

        x = self.patch_embeddings(x)
        for encoder in self.encoders:
            x = encoder(x)

        # Using the 0th index of the encoder output is mentioned in the equation 4 of the ViT paper, section 3.1.
        """
        The reason we are using the 0th index, 
        is that is this 0th index is the class token embedding we added to our input embeddings
        
        Since in bert language model, the encoder output can also be used in the decoder layers,
        we added an another class token embedding so that we can use this output for classification

        I hope this helps :)  
        """
        x = x[:, 0]

        x = self.mlp_head(x)

        return x


if __name__ == "__main__":
    img = torch.randn(1, 1, 28, 28)

    model = ViT(
        patch_size=7,
        image_size=28,
        in_channels=1,
        embed_dim=64,
        n_encoder=2,
        n_heads=8,
        n_classes=10,
    )

    # Printing the summary of the model
    summary(
        model,
        input_data=img,
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
        # depth=2,
    )
