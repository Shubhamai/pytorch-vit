"""Contains classes for Transformer Multi-head attention module."""

# Importing Libraries
from math import sqrt

import torch
import torch.nn as nn
from torchsummary import summary


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.):
        """Implementation of Scaled Dot Product Attention.

        The formula for computing the attention is written in the - `Attention is all you need` paper
        You will find it in to Page 3, section 3.2.1

        Args:
            dropout_p: Dropout probability for the attention.
        """
        super().__init__()

        self.dropout1 = nn.Dropout(dropout_p)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Computes the scaled dot product attention.

        We will use the formula from page 3, section 3.2.1, to compute the attention.

        Steps -
        1. Compute the dot product of the query and key tensors.
        2. Compute the softmax of the above dot product.
        3. Computes the dropout of the previous step.
        4. Compute the dot product of the value tensor and the above resustant tensor.

        Args:

            query: Tensor of shape [batch_size, seq_len, hidden_size]
            key: Tensor of shape [batch_size, seq_len, hidden_size]
            value: Tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Attention : Tensor of shape [batch_size, seq_len, hidden_size]
        """

        """
        Here we are -
        1. computing the dot product of the query and key tensors.
        2. dividing the above dot product by the square root of the key dimension.

        In short - 
        scores = query * key / sqrt(key.shape[-1])
        """
        scores = torch.divide(
            torch.bmm(query, key.transpose(1, 2)),
            sqrt(key.shape[-1]),  # bmm is batch matrix multiply
        )
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout1(weights)
        attention = torch.bmm(weights, value)

        return attention


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, attn_dropout_p: float = 0.):
        """Implementation of Single-head attention.

        Steps -
        1. Initialize the Linear layers for the query, key and value tensors.
        2. Initialize the Scaled Dot Product Attention module.

        Args:
            embed_dim: Dimension of the embedding.
            head_dim: Dimension of the head.
            attn_dropout_p: Dropout probability for the attention.
        """

        super().__init__()

        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p=attn_dropout_p)

        """
        Converting the embedding dimension into a much lower head dimension. This is done to decrease the computation load. 
        """
        self.query_linear = nn.Linear(embed_dim, head_dim)

        """
        One small thing to note is that in the attention paper, the output dim of key_linear and value_linear is the same. 
        But it doesn't have to be, it's not a rule. We can have the output dim according to our requirements.
        """
        self.key_linear = nn.Linear(embed_dim, head_dim)
        self.value_linear = nn.Linear(embed_dim, head_dim)

    def forward(self, embeddings) -> torch.Tensor:
        """Computes the single head attention

        Steps -
        1. Compute the query, key and value tensors from the embeddings. We do this by simple passing the same input into into 3 seeprate feedforward layers.
        2. Compute the scaled dot product attention from the above query, key and value tensors.

        Args:
            embeddings: Tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Attention : Tensor of shape [batch_size, seq_len, embed_dim]
        """

        query = self.query_linear(embeddings)
        key = self.key_linear(embeddings)
        value = self.value_linear(embeddings)

        attention = self.scaled_dot_product_attention(query, key, value)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, attn_dropout_p: float = 0.):
        """Implementation of Multi-head attention. Section 3.2.2 of Attention Paper

        Args:
            embed_dim: Dimension of the embedding.
            n_heads: Number of heads.
            attn_dropout: Dropout probability for the attention.
        """

        super().__init__()

        """
        Append the single attention head to the list according to the number of heads parameter in the class to form multi head attention.
        """
        head_dim = embed_dim // n_heads  # Note: Need mention
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim, attn_dropout_p) for _ in range(n_heads)]
        )

        """
        After concatenating the results of all the heads, we apply a linear layer to the concatenated results and bring the dimension back to original embedding dimension.
        """
        self.linear1 = nn.Linear(n_heads * head_dim, embed_dim)

    def forward(self, embedded_patches) -> torch.Tensor:
        """Computes the multi-head attention.

        Steps -
        1. Compute the attention for each head.
        2. Concatenate the results of all the heads.
        3. Apply the linear layer to the concatenated results.

        Args:
            embedded_patches: Tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Attention : Tensor of shape [batch_size, seq_len, embed_dim]
        """

        x = torch.cat([head(embedded_patches) for head in self.heads], dim=-1)

        x = self.linear1(x)

        return x


if __name__ == "__main__":

    # Defining the Multi-head attention module
    multi_head_attention = MultiHeadAttention(embed_dim=768, n_heads=2)

    # Defining the input tensor
    embeddings = torch.randn(1, 16, 768)

    # Computing the multi-head attention
    print(multi_head_attention(embeddings).shape)

    # Printing the summary of the model
    summary(
        multi_head_attention,
        input_data=embeddings,
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
        depth=2,
    )
