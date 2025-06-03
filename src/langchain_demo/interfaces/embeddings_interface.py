from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingsInterface(ABC):
    """
    Abstract base class for generating embeddings from text.
    """
    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the embeddings model.

        Args:
            model_name (str): The name of the model to use.
            **kwargs: Additional keyword arguments for model configuration.
        """
        self.model_name = model_name
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    
    @property
    @abstractmethod
    def emb_dim(self) -> int:
        """
        Returns the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings.
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts.
        Returns:
            List[torch.Tensor]: List of embeddings for the input texts.
        """
        pass