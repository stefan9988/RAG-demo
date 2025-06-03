from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import math
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddings(EmbeddingsInterface):
    """
    A class to generate sentence embeddings using Hugging Face models.
    """

    def __init__(self, model_name: str, 
                 api_key: Optional[str] = None, 
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 normalize_embeddings: bool = True,
                 max_length: int = 512):
        """
        Initializes the HuggingFaceEmbeddings class.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            api_key (Optional[str]): Hugging Face API key for private/gated models.
            device (Optional[str]): The device to use ('cuda', 'cpu'). 
            batch_size (int): The number of sentences to process in each batch.
                            Defaults to 32.
            normalize_embeddings (bool): Whether to normalize the embeddings to unit length (L2 norm).
                                        Defaults to True.
            max_length (int): Maximum sequence length for truncation. Defaults to 512.
        """
        super().__init__(model_name = model_name, 
                         api_key=api_key, 
                         device=device, 
                         batch_size=batch_size, 
                         normalize_embeddings=normalize_embeddings,
                         max_length=max_length)
        try:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Loading tokenizer: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
            logger.info(f"Loading model: {self.model_name}")
            
            self.model = AutoModel.from_pretrained(model_name, token=api_key)
            self.model.to(self.device)  
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise # Re-raise the exception after logging

    @property
    def emb_dim(self) -> int:
        """
        Returns the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings.
        """
        # Get the dimension from the model's config
        return self.model.config.hidden_size            
            
    def _mean_pooling(self,model_output, attention_mask):
        """
        Applies Mean Pooling to token embeddings.

        Args:
            model_output: Output from the transformer model.
            attention_mask: Attention mask to identify padding tokens.

        Returns:
            torch.Tensor: Sentence embeddings after mean pooling.
        """
        # Extract token embeddings (first element of model_output)
        token_embeddings = model_output[0]
        # Expand attention mask to match the shape of token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum embeddings where attention mask is 1, divide by the number of non-padding tokens
        # Clamp the sum of the mask to avoid division by zero
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def get_embeddings(
        self,
        texts: List[str],        
    ) -> np.ndarray:
        """
        Generates sentence embeddings for a list of texts using a Hugging Face model.

        Args:
            texts (List[str]): A list of sentences to embed.
                
        Returns:
            np.ndarray: A NumPy array of shape (num_sentences, embedding_dimension)
                        containing the sentence embeddings.
        """
        
        all_embeddings = []
        num_texts = len(texts)
        logger.info(f"Starting embedding generation for {num_texts} documents in batches of {self.batch_size}...")

        # --- Process in Batches ---
        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{math.ceil(num_texts/self.batch_size)}")

            # --- Tokenization ---
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length
            ).to(self.device) # Move tensors to the correct device

            # --- Inference ---
            with torch.no_grad(): # Disable gradient calculations
                model_output = self.model(**encoded_input)

            # --- Pooling ---
            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

            # --- Normalization ---
            if self.normalize_embeddings:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            # Move embeddings to CPU before storing to avoid accumulating GPU memory
            all_embeddings.append(batch_embeddings.cpu())

        # --- Combine Batches ---
        if not all_embeddings:
            return np.array([])

        final_embeddings_tensor = torch.cat(all_embeddings, dim=0)
        logger.info("Embeddings generation complete.")

        return final_embeddings_tensor.numpy()


    