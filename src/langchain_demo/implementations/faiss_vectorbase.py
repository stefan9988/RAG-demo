from langchain_demo.interfaces.vectorbase_interface import VectorbaseInterface
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
import faiss
from typing import List
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissVectorbase(VectorbaseInterface):
    DEFAULT_SAVE_DIR = "src/langchain_demo/faiss_vb"
    DEFAULT_INDEX_FILENAME = "faiss_index.index"
    DEFAULT_METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                 embeddings_model: EmbeddingsInterface, 
                 save_dir: str = DEFAULT_SAVE_DIR, 
                 index_filename: str = DEFAULT_INDEX_FILENAME,
                 metadata_filename: str = DEFAULT_METADATA_FILENAME):

        if not embeddings_model:
            raise ValueError("EmbeddingsInterface instance is required.")

        self.embeddings_model = embeddings_model
        self.index = None 
        self.stored_documents: list[str] = []

        # Construct full paths
        self.save_dir = save_dir
        self.index_path = os.path.join(save_dir, index_filename)
        self.metadata_path = os.path.join(save_dir, metadata_filename)

        # Ensure the save directory exists for future saves
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize a new FAISS index; load will overwrite this if files exist
        self._initialize_index()

    def _initialize_index(self):
        """Initializes a new FAISS index."""
        if self.embeddings_model:
            self.index = faiss.IndexFlatIP(self.embeddings_model.emb_dim)
            logger.info(f"Initialized new FAISS index with dimension {self.embeddings_model.emb_dim}.")
        else:
            self.index = None
            logger.warning("Embeddings model not available, FAISS index not initialized.")


    def add_documents(self, documents: list[str], document_name: str):
        if not self.embeddings_model:
            logger.error("Embeddings model not initialized. Cannot add documents.")
            raise ValueError("Embeddings model is not set.")
        if not self.index:
            logger.warning("Index not initialized. Attempting to initialize now.")
            self._initialize_index()
            if not self.index: 
                 raise RuntimeError("Failed to initialize FAISS index.")


        if not documents:
            logger.info("No documents provided to add.")
            return

        embeddings = self.embeddings_model.get_embeddings(documents)
        self.index.add(embeddings)
        self.stored_documents.extend(documents)

        if not hasattr(self, 'document_names'):
            self.document_names = []
    
        self.document_names.extend([document_name] * len(documents))

        logger.info(f"Added {len(documents)} documents. Index now contains {self.index.ntotal} vectors.")

    def search_index(self, queries: List[str], num_results: int = 5, threshold: float = 0.0, 
                 document_names: List[str] | None = None):
        if not self.embeddings_model:
            logger.error("Embeddings model not initialized. Cannot search documents.")
            raise ValueError("Embeddings model is not set.")
        if not self.index or self.index.ntotal == 0:
            logger.info("Index is empty. Cannot perform search.")
            return []

        query_embedding = self.embeddings_model.get_embeddings(queries)
        
        # If filtering by document names, we need to search more results initially
        # to account for filtering, then trim to the desired number
        search_k = self.index.ntotal if document_names else min(num_results, self.index.ntotal)
        
        if search_k == 0:
            logger.info("Effective k is 0, no search performed.")
            return []

        filter_msg = f" (filtering by documents: {document_names})" if document_names else ""
        logger.info(f"Searching for {num_results} most relevant documents with similarity threshold {threshold}{filter_msg}...")
        
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        
        for i in range(indices.shape[0]): 
            current_query_results = []   
            query_indices = indices[i]   
            query_similarities = distances[i] 

            for doc_index, sim in zip(query_indices, query_similarities): 
                if doc_index != -1 and sim >= threshold:
                    # Check if we have document names stored
                    if hasattr(self, 'document_names') and len(self.document_names) > doc_index:
                        doc_name = self.document_names[doc_index]
                        
                        # Apply document name filter if specified
                        if document_names and doc_name not in document_names:
                            continue  # Skip this document as it doesn't match the filter
                    else:
                        doc_name = None
                        # If filtering is requested but no document names are stored, skip
                        if document_names:
                            logger.warning("Document name filtering requested but no document names stored.")
                            continue
                    
                    document = self.stored_documents[doc_index]
                    result = {
                        "document id": int(doc_index), 
                        "document content": document, 
                        "similarity": float(sim),
                        "key": queries[i]
                    }
                    
                    # Add document name to result if available
                    if doc_name:
                        result["document_name"] = doc_name
                    
                    current_query_results.append(result)
                    
                    # Stop if we have enough results for this query
                    if len(current_query_results) >= num_results:
                        break
            
            results.append(current_query_results) 
            
        total_results = sum(len(query_results) for query_results in results)
        logger.info(f"Search completed. Found {total_results} results for {len(queries)} queries.")
        return results

    def save(self):
        """
        Saves the FAISS index and stored documents to the paths configured in the instance.
        """
        if not self.index:
            logger.warning("Index is not initialized or is empty. Nothing to save for FAISS index.")
            return
        if not self.embeddings_model:
            logger.error("Embeddings model not available. Cannot save embedding dimension.")
            raise ValueError("Embeddings model is required to save metadata.")

        os.makedirs(self.save_dir, exist_ok=True) 

        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}")

        metadata = {
            "stored_documents": self.stored_documents,
            "emb_dim": self.embeddings_model.emb_dim,
        }
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {self.metadata_path}")

    @classmethod
    def load(cls,
             embeddings_model: EmbeddingsInterface,
             load_dir: str = "src/langchain_demo/faiss_vb",
             index_filename: str = DEFAULT_INDEX_FILENAME,
             metadata_filename: str = DEFAULT_METADATA_FILENAME):
        """
        Loads the FAISS index and stored documents from the specified directory and filenames.

        Args:
            embeddings_model (EmbeddingsInterface): The embeddings model to use.
            load_dir (str): The directory path from where to load the files.
            index_filename (str): The name of the FAISS index file.
            metadata_filename (str): The name of the metadata file.

        Returns:
            FaissVectorbase: A new instance with loaded data.
        """
        if not embeddings_model:
            raise ValueError("An embeddings_model must be provided to load the FaissVectorbase.")

        actual_index_path = os.path.join(load_dir, index_filename)
        actual_metadata_path = os.path.join(load_dir, metadata_filename)

        if not os.path.exists(actual_index_path):
            logger.error(f"Index file not found: {actual_index_path}")
            raise FileNotFoundError(f"FAISS index file not found: {actual_index_path}")
        if not os.path.exists(actual_metadata_path):
            logger.error(f"Metadata file not found: {actual_metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {actual_metadata_path}")

        logger.info(f"Loading FAISS index from {actual_index_path}")
        loaded_index = faiss.read_index(actual_index_path)

        logger.info(f"Loading metadata from {actual_metadata_path}")
        with open(actual_metadata_path, "rb") as f:
            metadata = pickle.load(f)

        loaded_stored_documents = metadata["stored_documents"]
        saved_emb_dim = metadata["emb_dim"]

        if saved_emb_dim is not None and embeddings_model.emb_dim != saved_emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch! Saved: {saved_emb_dim}, Provided: {embeddings_model.emb_dim}"
            )
        if loaded_index.d != embeddings_model.emb_dim:
            raise ValueError(
                f"FAISS index dimension mismatch! Index: {loaded_index.d}, Provided Model: {embeddings_model.emb_dim}"
            )

        # Create instance, passing the paths it's being loaded from
        instance = cls(
            embeddings_model=embeddings_model,
            save_dir=load_dir,
            index_filename=index_filename,
            metadata_filename=metadata_filename
        )
        instance.index = loaded_index
        instance.stored_documents = loaded_stored_documents

        logger.info(f"FaissVectorbase loaded successfully. Index contains {instance.index.ntotal} vectors.")
        return instance