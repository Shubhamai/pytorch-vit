"""Implementing the encoder part of the Transformer with MLP block"""


# Importing Libraries
import torch
import torch.nn as nn

from vit.multi_head_attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout_p: float = 0.1):
        """Implementation of Multi-layer perceptron. Section 3.1, page 4 of ViT Paper

        Args:
            embed_dim: Dimension of the embedding.
            hidden_dim: Dimension of the hidden layer.
            dropout_p: Dropout probability.
        """
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(
            dropout_p
        )  # Added dropout layer, as mentioned in the section appendix B.1 in attention paper.

        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        """Computes the feed forward.

        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Attention : Tensor of shape [batch_size, seq_len, embed_dim]
        """

        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
    ):
        """Implements the Transformer Encoder. More info in Figure 1 and section 3.1 Encoder of the ViT paper and in attention paper.

        Args:
            embed_dim: Dimension of the embedding.
            n_heads: Number of heads.
            mlp_dropout_p: Dropout probability for the MLP block.
            attn_dropout_p: Dropout probability for the multi head attention block.
        """
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(
            embed_dim, n_heads, attn_dropout_p=attn_dropout_p
        )

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.multi_layer_preceptron = MLP(
            embed_dim,
            embed_dim
            * 4,  # A pattern is oberved in the ViT paper, table 1, page 5, that the hidden dimension is 4 times the embedding dimension.
            dropout_p=mlp_dropout_p,
        )

    def forward(self, embedded_patches) -> torch.Tensor:
        """Forward pass of the Encoder.

        Steps -
            1. Apply the layer norm to the input and then multi head attention.
            2. Add the original input to the output of the multi head attention.
            3. Apply the layer norm to the output of the previous layer and then feed forward.
            4. Add the output of the previous layer to the output of step 2.

        Args:
            embedded_patches: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Attention : Tensor of shape [batch_size, seq_len, embed_dim]
        """

        attention = self.multi_head_attention(self.layer_norm1(embedded_patches))
        x = embedded_patches + attention

        x = x + self.multi_layer_preceptron(self.layer_norm2(x))

        return x
