from langchain_demo.implementations.hugging_face_embeddings import HuggingFaceEmbeddings
from langchain_demo.implementations.openai_embeddings import OpenAIEmbeddings
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase
import os
from dotenv import load_dotenv

load_dotenv(override=True)
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL_NAME = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

embeddings_model = HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
        api_key=HUGGING_FACE_API_KEY)

# embeddings_model = OpenAIEmbeddings(
#     model_name=OPENAI_EMBEDDING_MODEL_NAME, 
#     api_key=OPENAI_API_KEY)
    
vector_db = FaissVectorbase(embeddings_model, index_filename="aa.index", metadata_filename="aa.pkl")

if __name__ == "__main__":
    
    docs_to_add = [
        "The weather today is sunny and warm.",
        "Yesterday it rained heavily all day long.",
        "Artificial intelligence is transforming many industries.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Many people enjoy sunny weather for outdoor activities.",
        "Deep learning is a subset of machine learning."
    ]
    vector_db.add_documents(docs_to_add)
    
    search_query = ["Information about France", "Info about AI"]
    similar_docs = vector_db.search_documents(search_query, num_results=3, threshold=0.1)
    print(f"\nQuery: '{search_query}'")
    print("Results:")
    for doc in similar_docs:
        print(f"- {doc}")