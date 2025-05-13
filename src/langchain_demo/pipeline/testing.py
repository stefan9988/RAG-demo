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

if __name__ == "__main__":
    # 1. Initialize the vectorbase
    
    # embeddings_model = HuggingFaceEmbeddings(
    #     model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
    #     api_key=HUGGING_FACE_API_KEY)
    embeddings_model = OpenAIEmbeddings(
        model_name=OPENAI_EMBEDDING_MODEL_NAME, 
        api_key=OPENAI_API_KEY)
    
    vector_db = FaissVectorbase(embeddings_model)

    # 2. Add some documents
    docs_to_add = [
        "The weather today is sunny and warm.",
        "Yesterday it rained heavily all day long.",
        "Artificial intelligence is transforming many industries.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Many people enjoy sunny weather for outdoor activities.",
        "Deep learning is a subset of machine learning."
    ]
    vector_db.add_documents(docs_to_add)

    print("\n--- Search Example 1 ---")
    # 3. Search for similar documents
    search_query_1 = "What is the weather like?"
    similar_docs_1 = vector_db.search_documents(search_query_1, num_results=3)
    print(f"\nQuery: '{search_query_1}'")
    print("Results:")
    for doc in similar_docs_1:
        print(f"- {doc}")

    print("\n--- Search Example 2 ---")
    search_query_2 = "Tell me about AI"
    similar_docs_2 = vector_db.search_documents(search_query_2, num_results=2)
    print(f"\nQuery: '{search_query_2}'")
    print("Results:")
    for doc in similar_docs_2:
        print(f"- {doc}")

    

    print("\n--- Add More Documents ---")
    vector_db.add_documents(["Paris is the capital of France."])

    print("\n--- Search Example 4 (After Adding More) ---")
    search_query_4 = "Information about France"
    similar_docs_4 = vector_db.search_documents(search_query_4, num_results=3)
    print(f"\nQuery: '{search_query_4}'")
    print("Results:")
    for doc in similar_docs_4:
        print(f"- {doc}")