import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil
import tempfile
import pickle
import numpy as np
import faiss

# Ensure this import path matches your project structure
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase

# --- Corrected Mock Embeddings Class ---
class MockEmbeddings(EmbeddingsInterface): # Or name it MockEmbeddingsModel if you prefer
    def __init__(self, dimension=10):
        self._dimension = dimension # Store the dimension internally
        self.call_count_get_embeddings = 0

    @property
    def emb_dim(self) -> int: # This is the crucial property
        return self._dimension

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        self.call_count_get_embeddings += 1
        embeddings = []
        for i, text in enumerate(texts):
            base_val = (i + 1) * 0.1
            # Ensure a consistent vector for a given text if needed, or unique like this
            emb_vector = np.array([base_val + (j * 0.01) for j in range(self._dimension)], dtype=np.float32)
            embeddings.append(emb_vector)
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1 and len(texts) == 1: # single text was passed
             embeddings_np = np.expand_dims(embeddings_np, axis=0)
        
        # Normalize embeddings as FAISS IndexFlatIP (inner product) works best with normalized vectors
        # for cosine similarity.
        norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / (norm + 1e-9) # Add epsilon to avoid division by zero
        return normalized_embeddings

class TestFaissVectorbase(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and a mock embeddings model for each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Ensure you are instantiating the corrected mock class here
        self.mock_embeddings_model = MockEmbeddings(dimension=5) # Use the class defined above

        self.index_filename = "test_faiss.index"
        self.metadata_filename = "test_metadata.pkl"
        
        # The following line was causing the error in your traceback because
        # self.mock_embeddings_model (or whatever you passed) didn't have .emb_dim
        self.vector_db = FaissVectorbase( # Renamed self.vb to self.vector_db to match your traceback
            embeddings_model=self.mock_embeddings_model,
            save_dir=self.test_dir,
            index_filename=self.index_filename,
            metadata_filename=self.metadata_filename
        )

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    # --- Test Cases ---
    # (Your test cases like test_add_and_search_documents, etc., will now use the correctly set up self.vector_db)

    def test_initialization_success(self):
        self.assertIsNotNone(self.vector_db.index, "FAISS index should be initialized.")
        # The FaissVectorbase will access self.mock_embeddings_model.emb_dim here
        self.assertEqual(self.vector_db.index.d, self.mock_embeddings_model.emb_dim, "Index dimension should match embeddings.")
        self.assertEqual(self.vector_db.stored_documents, [], "Stored documents should be empty initially.")
        self.assertTrue(os.path.exists(self.test_dir), "Save directory should be created.")
        self.assertEqual(self.vector_db.index_path, os.path.join(self.test_dir, self.index_filename))
        self.assertEqual(self.vector_db.metadata_path, os.path.join(self.test_dir, self.metadata_filename))

    def test_initialization_no_embeddings_model(self):
        with self.assertRaisesRegex(ValueError, "EmbeddingsInterface instance is required."):
            FaissVectorbase(embeddings_model=None, save_dir=self.test_dir)

    def test_add_documents_empty_list(self):
        initial_ntotal = self.vector_db.index.ntotal
        self.vector_db.add_documents([])
        self.assertEqual(self.vector_db.index.ntotal, initial_ntotal, "Adding empty list should not change index size.")
        self.assertEqual(len(self.vector_db.stored_documents), 0)

    def test_add_documents_success(self):
        docs = ["doc1", "doc2"]
        self.vector_db.add_documents(docs)
        self.assertEqual(self.vector_db.index.ntotal, len(docs), "Index should contain added documents.")
        self.assertEqual(self.vector_db.stored_documents, docs, "Stored documents should match added documents.")
        self.assertEqual(self.mock_embeddings_model.call_count_get_embeddings, 1)

    def test_add_documents_no_embeddings_model_runtime(self):
        original_model = self.vector_db.embeddings_model
        self.vector_db.embeddings_model = None
        with self.assertRaisesRegex(ValueError, "Embeddings model is not set."):
            self.vector_db.add_documents(["doc1"])
        self.vector_db.embeddings_model = original_model # Reset

    def test_add_documents_index_reinitialization(self):
        self.vector_db.index = None # Simulate index being None
        docs = ["doc1"]
        # Patch _initialize_index on the instance to check if it's called
        with patch.object(self.vector_db, '_initialize_index', wraps=self.vector_db._initialize_index) as mock_reinit:
            self.vector_db.add_documents(docs)
            mock_reinit.assert_called_once()
        self.assertIsNotNone(self.vector_db.index)
        self.assertEqual(self.vector_db.index.ntotal, 1)

    def test_search_documents_empty_index(self):
        results = self.vector_db.search_documents("query")
        self.assertEqual(results, [], "Search on empty index should return empty list.")

    def test_search_documents_no_embeddings_model_runtime(self):
        original_model = self.vector_db.embeddings_model
        self.vector_db.embeddings_model = None
        with self.assertRaisesRegex(ValueError, "Embeddings model is not set."):
            self.vector_db.search_documents("query")
        self.vector_db.embeddings_model = original_model # Reset

    def test_search_documents_success(self):
        docs = ["hello world", "test document", "another example"]
        self.vector_db.add_documents(docs)
        
        query = "hello world" 
        results = self.vector_db.search_documents(query, num_results=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "hello world")
        self.assertAlmostEqual(results[0][1], 1.0, places=5, msg="Similarity for identical doc should be ~1.0")

    def test_search_documents_num_results_and_threshold(self):
        docs = ["apple", "banana", "apricot", "avocado"]
        self.vector_db.add_documents(docs)
        # mock_embeddings_model.get_embeddings has been called once in add_documents

        # Search for "apple"
        results_k2 = self.vector_db.search_documents("apple", num_results=2)
        self.assertEqual(len(results_k2), 2)
        self.assertEqual(results_k2[0][0], "apple") # "apple" should be the most similar to itself

        results_k1_thresh_high = self.vector_db.search_documents("apple", num_results=1, threshold=0.9)
        self.assertEqual(len(results_k1_thresh_high), 1)
        self.assertEqual(results_k1_thresh_high[0][0], "apple")

        mock_faiss_index_search = MagicMock()
        # (Distances from IndexFlatIP are inner products; higher is more similar)
        # Mock returns: (distances_np_array, indices_np_array)
        # Example: query "apple" (index 0 in stored_documents)
        # Suppose search results are: doc0 (apple, sim 1.0), doc1 (banana, sim 0.5), doc2 (apricot, sim 0.2)
        mock_faiss_index_search.return_value = (
            np.array([[1.0, 0.5, 0.2]], dtype=np.float32), # similarities for 1 query
            np.array([[0, 1, 2]], dtype=np.int64)         # indices for 1 query
        )
        
        with patch.object(self.vector_db.index, 'search', mock_faiss_index_search):
            # Threshold 0.4: apple (1.0) and banana (0.5) should pass
            results_thresh_0_4 = self.vector_db.search_documents("apple", num_results=3, threshold=0.4)
            self.assertEqual(len(results_thresh_0_4), 2)
            self.assertTrue(any(r[0] == "apple" for r in results_thresh_0_4))
            self.assertTrue(any(r[0] == "banana" for r in results_thresh_0_4))

            # Threshold 0.6: only apple (1.0) should pass
            results_thresh_0_6 = self.vector_db.search_documents("apple", num_results=3, threshold=0.6)
            self.assertEqual(len(results_thresh_0_6), 1)
            self.assertEqual(results_thresh_0_6[0][0], "apple")

            # Threshold 1.1: nothing passes
            results_thresh_1_1 = self.vector_db.search_documents("apple", num_results=3, threshold=1.1)
            self.assertEqual(len(results_thresh_1_1), 0)

    def test_search_documents_k_is_zero(self):
        self.vector_db.add_documents(["doc1"])
        results = self.vector_db.search_documents("query", num_results=0)
        self.assertEqual(results, [])

    def test_save_and_load_cycle(self):
        docs = ["doc to save 1", "doc to save 2"]
        self.vector_db.add_documents(docs)
        original_ntotal = self.vector_db.index.ntotal
        original_docs = list(self.vector_db.stored_documents)

        self.vector_db.save()

        self.assertTrue(os.path.exists(self.vector_db.index_path))
        self.assertTrue(os.path.exists(self.vector_db.metadata_path))

        loaded_vb = FaissVectorbase.load(
            embeddings_model=self.mock_embeddings_model, # Use the same mock model type for loading
            load_dir=self.test_dir,
            index_filename=self.index_filename,
            metadata_filename=self.metadata_filename
        )

        self.assertIsNotNone(loaded_vb.index)
        self.assertEqual(loaded_vb.index.ntotal, original_ntotal)
        self.assertEqual(loaded_vb.stored_documents, original_docs)
        self.assertEqual(loaded_vb.index.d, self.mock_embeddings_model.emb_dim)

        results = loaded_vb.search_documents("doc to save 1", num_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc to save 1")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)
        
        self.assertEqual(loaded_vb.save_dir, self.test_dir)
        self.assertEqual(loaded_vb.index_path, os.path.join(self.test_dir, self.index_filename))
        self.assertEqual(loaded_vb.metadata_path, os.path.join(self.test_dir, self.metadata_filename))


    def test_save_no_index(self):
        self.vector_db.index = None
        with patch('langchain_demo.implementations.faiss_vectorbase.logger') as mock_logger:
             self.vector_db.save()
             mock_logger.warning.assert_called_with("Index is not initialized or is empty. Nothing to save for FAISS index.")
        self.assertFalse(os.path.exists(self.vector_db.index_path))
        self.assertFalse(os.path.exists(self.vector_db.metadata_path))


    def test_save_no_embeddings_model_runtime(self):
        self.vector_db.add_documents(["doc1"]) 
        original_model = self.vector_db.embeddings_model
        self.vector_db.embeddings_model = None
        with self.assertRaisesRegex(ValueError, "Embeddings model is required to save metadata."):
            self.vector_db.save()
        self.vector_db.embeddings_model = original_model

    def test_load_no_embeddings_model_provided(self):
        with self.assertRaisesRegex(ValueError, "An embeddings_model must be provided to load the FaissVectorbase."):
            FaissVectorbase.load(embeddings_model=None, load_dir=self.test_dir)

    def test_load_file_not_found(self):
        with self.assertRaisesRegex(FileNotFoundError, "FAISS index file not found"):
            FaissVectorbase.load(embeddings_model=self.mock_embeddings_model, load_dir=self.test_dir)

        # Create a dummy index file, but no metadata file
        # Need an actual index object to write, so let's quickly get one from a temp vb
        temp_vb = FaissVectorbase(self.mock_embeddings_model, save_dir=self.test_dir, index_filename="temp_idx.idx")
        temp_vb.add_documents(["dummy"]) # ensure index is not empty
        faiss.write_index(temp_vb.index, os.path.join(self.test_dir, self.index_filename))
        
        with self.assertRaisesRegex(FileNotFoundError, "Metadata file not found"):
            FaissVectorbase.load(
                embeddings_model=self.mock_embeddings_model, 
                load_dir=self.test_dir,
                index_filename=self.index_filename # specify the one we just created
                )


    def test_load_embedding_dimension_mismatch_metadata(self):
        self.vector_db.add_documents(["doc for dim mismatch"])
        self.vector_db.save() # Saves with mock_embeddings_model.emb_dim (5)

        mismatched_embeddings_model = MockEmbeddings(dimension=10) 
        
        with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch!"):
            FaissVectorbase.load(
                embeddings_model=mismatched_embeddings_model,
                load_dir=self.test_dir,
                index_filename=self.index_filename,
                metadata_filename=self.metadata_filename
            )

    def test_load_faiss_index_dimension_mismatch(self):
        # self.vector_db uses mock_embeddings_model (dim 5)
        self.vector_db.add_documents(["doc1"])
        self.vector_db.save() # Index file saved with dimension 5

        # New model with a different dimension
        mismatched_dim_model = MockEmbeddings(dimension=self.mock_embeddings_model.emb_dim + 1) # e.g., dim 6

        # To isolate the FAISS index dimension check, we need the metadata check to pass.
        # So, we'll modify the saved metadata to match the mismatched_dim_model's dimension.
        metadata_path_to_modify = os.path.join(self.test_dir, self.metadata_filename)
        with open(metadata_path_to_modify, "rb") as f:
            metadata = pickle.load(f)
        
        metadata["emb_dim"] = mismatched_dim_model.emb_dim # "Trick" metadata to match new model
        
        with open(metadata_path_to_modify, "wb") as f:
            pickle.dump(metadata, f)
        
        # Now, metadata dim matches mismatched_dim_model, but actual FAISS file dim (5) does not.
        with self.assertRaisesRegex(ValueError, "FAISS index dimension mismatch!"):
            FaissVectorbase.load(
                embeddings_model=mismatched_dim_model,
                load_dir=self.test_dir,
                index_filename=self.index_filename,
                metadata_filename=self.metadata_filename
            )

    def test_default_filenames_used(self):
        vb_default = FaissVectorbase(
            embeddings_model=self.mock_embeddings_model,
            save_dir=self.test_dir
        )
        self.assertEqual(vb_default.index_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_INDEX_FILENAME))
        self.assertEqual(vb_default.metadata_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_METADATA_FILENAME))

        docs = ["default test"]
        vb_default.add_documents(docs)
        vb_default.save()

        loaded_vb_default = FaissVectorbase.load(
            embeddings_model=self.mock_embeddings_model,
            load_dir=self.test_dir
            # Intentionally not providing index_filename and metadata_filename to test defaults
        )
        self.assertEqual(len(loaded_vb_default.stored_documents), 1)
        self.assertEqual(loaded_vb_default.stored_documents[0], "default test")
        self.assertEqual(loaded_vb_default.index_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_INDEX_FILENAME))
        self.assertEqual(loaded_vb_default.metadata_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_METADATA_FILENAME))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)