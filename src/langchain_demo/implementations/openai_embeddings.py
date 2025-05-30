from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
from typing import List, Optional
import logging
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIEmbeddings(EmbeddingsInterface):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize OpenAI embeddings with the specified model name and API key.
        Args:
            model_name (str): The name of the OpenAI model to use for embeddings.
            api_key (str, optional): The API key for OpenAI. Defaults to None.            
        """
        super().__init__(model_name=model_name, api_key=api_key)
        
        self.client = OpenAI(api_key=self.api_key)
        self._emb_dim = self.get_embeddings(["test"]).shape[1]
    
    @property
    def emb_dim(self) -> int:
        return self._emb_dim
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using OpenAI's API.
        Args:
            texts (List[str]): List of input texts to generate embeddings for.
        Returns:
            List[List[float]]: List of embeddings for the input texts.
        """
        logger.info(f"Generating embeddings for {len(texts)} documents using model {self.model_name}.")
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )    
        
        embeddings = [response.data[i].embedding for i in range(len(response.data))]
        embeddings_array = np.array(embeddings).astype('float32')
        
        return embeddings_array
        
