from langchain_demo.implementations.hugging_face_embeddings import HuggingFaceEmbeddings
from langchain_demo.implementations.openai_embeddings import OpenAIEmbeddings
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase
from langchain_demo.implementations.langchain_textsplitter import LangchainTextSplitter
from langchain_demo.pipeline.helper_functions import get_info_to_extract, create_similar_docs_batches

import os
from langchain_community.document_loaders import PyPDFLoader
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
    
# vector_db = FaissVectorbase(embeddings_model, index_filename="hf.index", metadata_filename="hf.pkl")
# splitter = LangchainTextSplitter()

if __name__ == "__main__":
    
    # filename = "data/Policy 01-Nov-2024-3607.pdf"
    # loader = PyPDFLoader(filename)
    # pages = [page for page in loader.lazy_load()]
    
    # documents = splitter.split_text(pages, chunk_size=500, chunk_overlap=50)
    
    # docs_to_add = [doc.page_content for doc in documents]
    
    # vector_db.add_documents(docs_to_add)
    # vector_db.save()
    
    vector_db= FaissVectorbase.load(embeddings_model, index_filename="hf.index", metadata_filename="hf.pkl")
    
    info_to_extract = get_info_to_extract("data/info_to_extract.json")
    
    similar_docs = vector_db.search_documents(info_to_extract, num_results=3, threshold=0.1)
    similar_docs_batches = create_similar_docs_batches(similar_docs, batch_size=8)
    