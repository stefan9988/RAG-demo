from abc import ABC, abstractmethod

class VectorbaseInterface(ABC):
    @abstractmethod
    def add_documents(self, documents):
        pass
    @abstractmethod
    def search_documents(self, query, num_results=5, threshold=0):
        pass
    
