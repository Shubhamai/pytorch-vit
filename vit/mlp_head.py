"""Implementing  the MLP Head for Vision Transformer
"""

# Importing Libraries
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        """Implementation of the MLP Head for Vision Transformer.

        Args:
            embed_dim (int): Dimension of the embedding.
            n_classes (int): Number of classes in the output.
        """

        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Computes the feed forward

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes)
        """

        x = self.layernorm1(x)
        x = self.linear1(x)
        return x
