from langchain_demo.interfaces.vectorbase_interface import VectorbaseInterface
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
import faiss
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissVectorbase(VectorbaseInterface):
    def __init__(self, embeddings_model:EmbeddingsInterface = None):
        self.embeddings_model = embeddings_model
        self.embeddings_model_name = embeddings_model.model_name 
        self.index = faiss.IndexFlatIP(self.embeddings_model.emb_dim)
        self.stored_documents = []
        
    def add_documents(self, documents: list[str]):
        """
        Adds documents to the vector store.

        Args:
            documents (list[str]): A list of text documents to add.
        """
        if not documents:
            logger.info("No documents provided to add.")
            return

        # 1. Get the embeddings for the documents
        embeddings = self.embeddings_model.get_embeddings(documents)

        # 2. Add embeddings to the FAISS index
        self.index.add(embeddings)

        # 3. Store the original documents
        self.stored_documents.extend(documents)

        logger.info(f"Added {len(documents)} documents. Index now contains {self.index.ntotal} vectors.")
    
    def search_documents(self, query, num_results=5):
        """
        Searches for documents similar to the query.

        Args:
            query (str): The search query text.
            num_results (int): The maximum number of similar documents to return.

        Returns:
            list[str]: A list of the most similar documents found.
        """
        if self.index.ntotal == 0:
            logger.info("Index is empty. Cannot perform search.")
            return []

        # 1. Get the embedding for the query
        query_embedding = self.embeddings_model.get_embeddings([query])

        # Ensure k is not greater than the number of vectors in the index
        k = min(num_results, self.index.ntotal)
        logger.info(f"Searching for {k} most relevant documents...")

        # 2. Search the FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # 3. Retrieve the corresponding documents
        results = []
        if indices.size > 0:
            query_indices = indices[0]
            query_distances = distances[0]
            
            assert len(query_indices) == len(query_distances), "Indices and distances lengths mismatch"

            for doc_index, distance in zip(query_indices, query_distances):
                # FAISS uses -1 for invalid indices (e.g., if k > number of docs)
                if doc_index != -1:
                    document = self.stored_documents[doc_index]
                    results.append((document, float(distance)))

        logger.info(f"Found {len(results)} similar documents.")
        
        return results