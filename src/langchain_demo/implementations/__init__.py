import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_demo.implementations.hugging_face_embeddings import HuggingFaceEmbeddings
from langchain_demo.implementations.openai_embeddings import OpenAIEmbeddings
from langchain_demo.implementations.openai_api_call import OpenAIAPICall
from langchain_demo.implementations.bedrock_api_call import BedrockAPICall
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase
from langchain_demo.interfaces.embeddings_interface import EmbeddingsInterface
from langchain_demo.config import (
    HUGGINGFACE_EMBEDDING_MODEL_NAME, 
    OPENAI_EMBEDDING_MODEL_NAME,
    OPENAI_MODEL, 
    BEDROCK_MODEL,
    SYSTEM_PROMPT, 
    TEMPERATURE, 
    MAX_TOKENS,
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    SEPARATORS, 
    INDEX_FILENAME, 
    METADATA_FILENAME, 
    VB_SAVE_DIR
)

def get_llm_api(llm_provider):
    load_dotenv(override=True)
    
    if llm_provider == 'openai':
        return OpenAIAPICall(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model=OPENAI_MODEL,
            system_prompt=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    elif llm_provider == 'bedrock':
        return BedrockAPICall(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            api_key=None,  
            messages=None,
            region=os.getenv("AWS_REGION"),
            model=BEDROCK_MODEL,
            system_prompt=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

def get_embeddings_model(embedding_provider):
    load_dotenv(override=True)
    
    if embedding_provider == 'huggingface':
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
            api_key=os.getenv("HUGGING_FACE_API_KEY")
        )
    elif embedding_provider == 'openai':
        return OpenAIEmbeddings(
            model_name=OPENAI_EMBEDDING_MODEL_NAME, 
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
def get_vectorbase(vectorbase_provider:str, embeddings_model:EmbeddingsInterface):
    if vectorbase_provider == 'faiss':
        return FaissVectorbase(embeddings_model,
                            save_dir=VB_SAVE_DIR,
                            index_filename=INDEX_FILENAME,
                            metadata_filename=METADATA_FILENAME)
    else:
        raise ValueError(f"Unsupported vectorbase provider: {vectorbase_provider}")
    
def get_text_splitter(text_splitter:str):
    if text_splitter =='recursive_character':
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
    else:
        raise ValueError(f"Unsupported text splitter: {text_splitter}")