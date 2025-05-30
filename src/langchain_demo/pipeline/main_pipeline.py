from langchain_demo.implementations.hugging_face_embeddings import HuggingFaceEmbeddings
from langchain_demo.implementations.openai_embeddings import OpenAIEmbeddings
from langchain_demo.implementations.faiss_vectorbase import FaissVectorbase
from langchain_demo.implementations.langchain_textsplitter import LangchainTextSplitter
from langchain_demo.implementations.openai_api_call import OpenAIAPICall
from langchain_demo.implementations.bedrock_api_call import BedrockAPICall

from langchain_demo.pipeline.helper_functions import get_info_to_extract, create_similar_docs_batches 
from langchain_demo.pipeline.helper_functions import create_key_description_pairs, extract_unique_keys_per_batch
from langchain_demo.pipeline.helper_functions import join_document_content_per_batch, create_messages
from langchain_demo.pipeline.helper_functions import create_extracted_info_dict, ensure_all_info_keys_present
from langchain_demo.pipeline.helper_functions import parallel_api_calls, process_api_extractions

from langchain_demo.config import SYSTEM_PROMPT, OPENAI_MODEL, BEDROCK_MODEL,TEMPERATURE, MAX_TOKENS 
from langchain_demo.config import HUGGINGFACE_EMBEDDING_MODEL_NAME, OPENAI_EMBEDDING_MODEL_NAME
from langchain_demo.config import VB_NUM_RESULTS, VB_SEARCH_THRESHOLD, INFO_EXTRACTION_BATCH_SIZE

import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv(override=True)
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

embeddings_model = HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
        api_key=HUGGING_FACE_API_KEY)

# embeddings_model = OpenAIEmbeddings(
#     model_name=OPENAI_EMBEDDING_MODEL_NAME, 
#     api_key=OPENAI_API_KEY)
    
# vector_db = FaissVectorbase(embeddings_model, index_filename="hf.index", metadata_filename="hf.pkl")
# splitter = LangchainTextSplitter()

# openai_api = OpenAIAPICall(api_key=OPENAI_API_KEY, model=MODEL,
#                         system_prompt= SYSTEM_PROMPT,temperature=TEMPERATURE,
#                         max_tokens=MAX_TOKENS)

bedrock_api = BedrockAPICall(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    api_key=None,  
    messages= None,
    region=AWS_REGION,
    model=BEDROCK_MODEL,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

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
    
    relevant_docs = vector_db.search_documents(info_to_extract, 
                                               num_results=VB_NUM_RESULTS, 
                                               threshold=VB_SEARCH_THRESHOLD)
    key_description_pairs = create_key_description_pairs(relevant_docs)
    relevant_docs_batches = create_similar_docs_batches(key_description_pairs, 
                                                        batch_size=INFO_EXTRACTION_BATCH_SIZE)
    
    doc_content_batches = join_document_content_per_batch(relevant_docs_batches)
    info_to_extract_batches = extract_unique_keys_per_batch(relevant_docs_batches)
    messages = create_messages(doc_content_batches, info_to_extract_batches)
    
    # extracted_info,usage = bedrock_api.call(message=messages[0]) 
    api_responses = asyncio.run(parallel_api_calls(bedrock_api, messages))
    #TODO: fix Race condition issue with parallel_api_calls
    extracted_info_dict = process_api_extractions(api_responses)
    
    extracted_info_dict = ensure_all_info_keys_present(extracted_info_dict, sum(info_to_extract_batches, []))
    print(extracted_info_dict)
    
    
    

    
    
        
        
    
    