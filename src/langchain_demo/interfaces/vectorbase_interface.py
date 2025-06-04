from abc import ABC, abstractmethod
from typing import List
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface

class VectorbaseInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], document_name: str):
        pass
    @abstractmethod
    def search_index(self, 
                     queries: List[str], 
                     num_results:int = 5,
                     threshold:float = 0,
                     document_names: List[str] | None = None):
        pass
    @abstractmethod
    def save(self):
        pass
    @abstractmethod
    def load(cls,
             embeddings_model: EmbeddingsInterface,
             load_dir: str,
             index_filename: str,
             metadata_filename: str):
        pass
