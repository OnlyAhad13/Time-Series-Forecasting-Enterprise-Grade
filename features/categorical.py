import torch
import torch.nn as nn
from typing import Optional

class CategoryEmbedding(nn.Module):
    """
    Embedding layer for categorical features
    """

    def __init__(
        self, 
        num_categories: int, 
        embedding_dim: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) or batch_size
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim)
        """
        return self.embedding(x)