from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingsInterface(ABC):
    """
    Abstract base class for generating embeddings from text.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initializes the embeddings model.

        Args:
            model_name (str): The name of the model to use.
            api_key (Optional[str]): API key for accessing the model.            
        """
        self.model_name = model_name
        self.api_key = api_key
    
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
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts.
        Returns:
            List[torch.Tensor]: List of embeddings for the input texts.
        """
        pass