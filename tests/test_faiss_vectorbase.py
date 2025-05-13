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
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase # Adjusted path

# --- Mock Embeddings Class (remains the same) ---
class MockEmbeddings(EmbeddingsInterface):
    def __init__(self, dimension=10):
        self._dimension = dimension
        self.call_count_get_embeddings = 0

    @property
    def emb_dim(self) -> int:
        return self._dimension

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        self.call_count_get_embeddings += 1
        embeddings = []
        for i, text in enumerate(texts):
            base_val = (i + 1) * 0.1
            emb_vector = np.array([base_val + (j * 0.01) for j in range(self._dimension)], dtype=np.float32)
            embeddings.append(emb_vector)
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1 and len(texts) == 1:
             embeddings_np = np.expand_dims(embeddings_np, axis=0)
        
        norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / (norm + 1e-9)
        return normalized_embeddings

class TestFaissVectorbase(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_embeddings_model = MockEmbeddings(dimension=5)
        self.index_filename = "test_faiss.index"
        self.metadata_filename = "test_metadata.pkl"
        self.vector_db = FaissVectorbase(
            embeddings_model=self.mock_embeddings_model,
            save_dir=self.test_dir,
            index_filename=self.index_filename,
            metadata_filename=self.metadata_filename
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization_success(self):
        self.assertIsNotNone(self.vector_db.index)
        self.assertEqual(self.vector_db.index.d, self.mock_embeddings_model.emb_dim)
        self.assertEqual(self.vector_db.stored_documents, [])
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(self.vector_db.index_path, os.path.join(self.test_dir, self.index_filename))
        self.assertEqual(self.vector_db.metadata_path, os.path.join(self.test_dir, self.metadata_filename))

    def test_initialization_no_embeddings_model(self):
        with self.assertRaisesRegex(ValueError, "EmbeddingsInterface instance is required."):
            FaissVectorbase(embeddings_model=None, save_dir=self.test_dir)

    def test_add_documents_empty_list(self):
        initial_ntotal = self.vector_db.index.ntotal
        self.vector_db.add_documents([])
        self.assertEqual(self.vector_db.index.ntotal, initial_ntotal)
        self.assertEqual(len(self.vector_db.stored_documents), 0)

    def test_add_documents_success(self):
        docs = ["doc1", "doc2"]
        self.vector_db.add_documents(docs)
        self.assertEqual(self.vector_db.index.ntotal, len(docs))
        self.assertEqual(self.vector_db.stored_documents, docs)
        self.assertEqual(self.mock_embeddings_model.call_count_get_embeddings, 1)

    def test_add_documents_no_embeddings_model_runtime(self):
        original_model = self.vector_db.embeddings_model
        self.vector_db.embeddings_model = None
        with self.assertRaisesRegex(ValueError, "Embeddings model is not set."):
            self.vector_db.add_documents(["doc1"])
        self.vector_db.embeddings_model = original_model

    def test_add_documents_index_reinitialization(self):
        self.vector_db.index = None
        docs = ["doc1"]
        with patch.object(self.vector_db, '_initialize_index', wraps=self.vector_db._initialize_index) as mock_reinit:
            self.vector_db.add_documents(docs)
            mock_reinit.assert_called_once()
        self.assertIsNotNone(self.vector_db.index)
        self.assertEqual(self.vector_db.index.ntotal, 1)

    def test_search_documents_empty_index(self):
        # If index is empty, method returns []
        results = self.vector_db.search_documents(["query"])
        self.assertEqual(results, [])

    def test_search_documents_no_embeddings_model_runtime(self):
        original_model = self.vector_db.embeddings_model
        self.vector_db.embeddings_model = None
        with self.assertRaisesRegex(ValueError, "Embeddings model is not set."):
            self.vector_db.search_documents(["query"])
        self.vector_db.embeddings_model = original_model

    def test_search_documents_success_single_query(self):
        docs = ["hello world", "test document", "another example"]
        self.vector_db.add_documents(docs)
        
        query = ["hello world"] 
        results = self.vector_db.search_documents(query, num_results=1)
        
        self.assertEqual(len(results), 1, "Should return a list of results for the single query")
        self.assertTrue(isinstance(results[0], list), "Each query's result should be a list")
        self.assertEqual(len(results[0]), 1, "The single query should find 1 document")
        
        doc_result = results[0][0]
        self.assertEqual(doc_result[0], "hello world")
        self.assertAlmostEqual(doc_result[1], 1.0, places=5, msg="Similarity for identical doc should be ~1.0")

    def test_search_documents_num_results_and_threshold_single_query(self):
        docs = ["apple", "banana", "apricot", "avocado"]
        self.vector_db.add_documents(docs)

        # Search for ["apple"]
        results_k2 = self.vector_db.search_documents(["apple"], num_results=2)
        self.assertEqual(len(results_k2), 1) # One list for the one query
        self.assertEqual(len(results_k2[0]), 2) # Two results for that query
        self.assertEqual(results_k2[0][0][0], "apple")

        results_k1_thresh_high = self.vector_db.search_documents(["apple"], num_results=1, threshold=0.9)
        self.assertEqual(len(results_k1_thresh_high), 1)
        self.assertEqual(len(results_k1_thresh_high[0]), 1)
        self.assertEqual(results_k1_thresh_high[0][0][0], "apple")

        # Mocking faiss.Index.search (returns 2D arrays for distances and indices)
        mock_faiss_index_search = MagicMock()
        # For a single query in the batch ["apple"], FAISS would return something like:
        # distances: np.array([[1.0, 0.5, 0.2]]) (1 row for 1 query)
        # indices: np.array([[0, 1, 2]]) (1 row for 1 query)
        # where 0 is "apple", 1 is "banana", 2 is "apricot" in self.vector_db.stored_documents
        mock_faiss_index_search.return_value = (
            np.array([[1.0, 0.5, 0.2]], dtype=np.float32), 
            np.array([[0, 1, 2]], dtype=np.int64)      
        )
        
        with patch.object(self.vector_db.index, 'search', mock_faiss_index_search):
            # Threshold 0.4: apple (1.0) and banana (0.5) should pass for the first query
            results_thresh_0_4 = self.vector_db.search_documents(["apple"], num_results=3, threshold=0.4)
            self.assertEqual(len(results_thresh_0_4), 1) # Results for one query
            self.assertEqual(len(results_thresh_0_4[0]), 2) # Two documents meet criteria
            self.assertTrue(any(r[0] == "apple" for r in results_thresh_0_4[0]))
            self.assertTrue(any(r[0] == "banana" for r in results_thresh_0_4[0]))

            # Threshold 0.6: only apple (1.0) should pass
            results_thresh_0_6 = self.vector_db.search_documents(["apple"], num_results=3, threshold=0.6)
            self.assertEqual(len(results_thresh_0_6), 1)
            self.assertEqual(len(results_thresh_0_6[0]), 1)
            self.assertEqual(results_thresh_0_6[0][0][0], "apple")

            # Threshold 1.1: nothing passes
            results_thresh_1_1 = self.vector_db.search_documents(["apple"], num_results=3, threshold=1.1)
            self.assertEqual(len(results_thresh_1_1), 1)
            self.assertEqual(len(results_thresh_1_1[0]), 0) # Empty list of results for the query

    def test_search_documents_no_results_above_threshold(self):
        self.vector_db.add_documents(["doc1"])
        # Search for "doc1", but with a threshold so high nothing matches
        # (except the doc itself if similarity is 1.0, but test with threshold > 1.0)
        results = self.vector_db.search_documents(["doc1"], threshold=1.1)
        self.assertEqual(len(results), 1)       # One list for the one query
        self.assertEqual(len(results[0]), 0)    # No documents found for that query

    def test_search_documents_k_is_zero(self):
        self.vector_db.add_documents(["doc1"])
        # If k becomes 0 due to num_results=0, method returns []
        results = self.vector_db.search_documents(["query"], num_results=0)
        self.assertEqual(results, [])


    def test_search_documents_batch_queries(self):
        docs = ["red apple", "green banana", "yellow lemon", "blue berry"]
        self.vector_db.add_documents(docs)

        queries = ["fruit with color red", "yellow citrus fruit"]
        
        # Let's assume "fruit with color red" is most similar to "red apple"
        # And "yellow citrus fruit" is most similar to "yellow lemon"
        
        # Mock the search to control output for two queries
        # Query 1 ("fruit with color red") -> finds "red apple" (index 0)
        # Query 2 ("yellow citrus fruit") -> finds "yellow lemon" (index 2), "green banana" (index 1)
        mock_faiss_index_search = MagicMock()
        mock_faiss_index_search.return_value = (
            np.array([[0.9, 0.3], [0.8, 0.4]], dtype=np.float32), # Distances for 2 queries, k=2
            np.array([[0, 3], [2, 1]], dtype=np.int64)            # Indices for 2 queries, k=2
        )
        # stored_documents: ["red apple", "green banana", "yellow lemon", "blue berry"]
        # Query 1: idx 0 ("red apple", sim 0.9), idx 3 ("blue berry", sim 0.3)
        # Query 2: idx 2 ("yellow lemon", sim 0.8), idx 1 ("green banana", sim 0.4)

        with patch.object(self.vector_db.index, 'search', mock_faiss_index_search):
            results = self.vector_db.search_documents(queries, num_results=2, threshold=0.35)
            
            self.assertEqual(len(results), 2, "Should be results for 2 queries")

            # Results for "fruit with color red" (query 1)
            self.assertEqual(len(results[0]), 1) # Only "red apple" (0.9) should pass threshold 0.35
            self.assertEqual(results[0][0][0], "red apple")
            self.assertAlmostEqual(results[0][0][1], 0.9, places=5)

            # Results for "yellow citrus fruit" (query 2)
            self.assertEqual(len(results[1]), 2) # "yellow lemon" (0.8) and "green banana" (0.4) should pass
            # Order might not be guaranteed by FAISS after our filtering, so check presence
            found_docs_q2 = [res[0] for res in results[1]]
            self.assertIn("yellow lemon", found_docs_q2)
            self.assertIn("green banana", found_docs_q2)


    def test_save_and_load_cycle(self):
        docs = ["doc to save 1", "doc to save 2"]
        self.vector_db.add_documents(docs)
        original_ntotal = self.vector_db.index.ntotal
        original_docs = list(self.vector_db.stored_documents)

        self.vector_db.save()

        self.assertTrue(os.path.exists(self.vector_db.index_path))
        self.assertTrue(os.path.exists(self.vector_db.metadata_path))

        loaded_vb = FaissVectorbase.load(
            embeddings_model=self.mock_embeddings_model,
            load_dir=self.test_dir,
            index_filename=self.index_filename,
            metadata_filename=self.metadata_filename
        )

        self.assertIsNotNone(loaded_vb.index)
        self.assertEqual(loaded_vb.index.ntotal, original_ntotal)
        self.assertEqual(loaded_vb.stored_documents, original_docs)
        self.assertEqual(loaded_vb.index.d, self.mock_embeddings_model.emb_dim)

        results = loaded_vb.search_documents(["doc to save 1"], num_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[0][0][0], "doc to save 1")
        self.assertAlmostEqual(results[0][0][1], 1.0, places=5)
        
        self.assertEqual(loaded_vb.save_dir, self.test_dir)
        self.assertEqual(loaded_vb.index_path, os.path.join(self.test_dir, self.index_filename))
        self.assertEqual(loaded_vb.metadata_path, os.path.join(self.test_dir, self.metadata_filename))

    # --- Other tests (save, load error handling, defaults) remain largely the same ---
    # --- as they don't directly call search_documents in a way that its output structure matters ---
    # --- or they test conditions before search could even happen.                       ---

    def test_save_no_index(self):
        self.vector_db.index = None
        with patch('langchain_demo.implementations.faiss_vectorbase.logger') as mock_logger: # adjust path if needed
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
        temp_vb_for_index = FaissVectorbase(self.mock_embeddings_model, save_dir=self.test_dir, index_filename="temp_idx.idx")
        temp_vb_for_index.add_documents(["dummy"])
        faiss.write_index(temp_vb_for_index.index, os.path.join(self.test_dir, self.index_filename))
        
        with self.assertRaisesRegex(FileNotFoundError, "Metadata file not found"):
            FaissVectorbase.load(
                embeddings_model=self.mock_embeddings_model, 
                load_dir=self.test_dir,
                index_filename=self.index_filename
                )

    def test_load_embedding_dimension_mismatch_metadata(self):
        self.vector_db.add_documents(["doc for dim mismatch"])
        self.vector_db.save() 

        mismatched_embeddings_model = MockEmbeddings(dimension=10) 
        
        with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch!"):
            FaissVectorbase.load(
                embeddings_model=mismatched_embeddings_model,
                load_dir=self.test_dir,
                index_filename=self.index_filename,
                metadata_filename=self.metadata_filename
            )

    def test_load_faiss_index_dimension_mismatch(self):
        self.vector_db.add_documents(["doc1"])
        self.vector_db.save() 

        mismatched_dim_model = MockEmbeddings(dimension=self.mock_embeddings_model.emb_dim + 1)

        metadata_path_to_modify = os.path.join(self.test_dir, self.metadata_filename)
        with open(metadata_path_to_modify, "rb") as f:
            metadata = pickle.load(f)
        metadata["emb_dim"] = mismatched_dim_model.emb_dim
        with open(metadata_path_to_modify, "wb") as f:
            pickle.dump(metadata, f)
        
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
        )
        self.assertEqual(len(loaded_vb_default.stored_documents), 1)
        self.assertEqual(loaded_vb_default.stored_documents[0], "default test")
        self.assertEqual(loaded_vb_default.index_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_INDEX_FILENAME))
        self.assertEqual(loaded_vb_default.metadata_path, os.path.join(self.test_dir, FaissVectorbase.DEFAULT_METADATA_FILENAME))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)