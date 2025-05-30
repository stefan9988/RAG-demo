from abc import ABC, abstractmethod
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface

class VectorbaseInterface(ABC):
    @abstractmethod
    def add_documents(self, documents):
        pass
    @abstractmethod
    def search_documents(self, query, num_results=5, threshold=0):
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
