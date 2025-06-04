from langchain_demo.pipeline.helper_functions import (get_info_to_extract, create_similar_docs_batches,
        create_key_description_pairs, extract_unique_keys_per_batch,join_document_content_per_batch, 
        create_messages, ensure_all_info_keys_present,parallel_api_calls, process_api_extractions)

from langchain_demo.config import VB_NUM_RESULTS, VB_SEARCH_THRESHOLD, INFO_EXTRACTION_BATCH_SIZE
from langchain_demo.implementations import get_llm_api, get_embeddings_model, get_text_splitter, get_vectorbase

import asyncio
from langchain_community.document_loaders import PyPDFLoader

embeddings_model = get_embeddings_model('huggingface')
llm_api = get_llm_api('openai')
text_splitter = get_text_splitter('recursive_character')
vector_db = get_vectorbase('faiss', embeddings_model)

if __name__ == "__main__":
    
    filename = "data/Policy 01-Nov-2024-3607.pdf"
    loader = PyPDFLoader(filename)
    pages = [page for page in loader.lazy_load()]
    
    documents = text_splitter.split_documents(pages)
    
    docs_to_add = [doc.page_content for doc in documents]
    
    vector_db.add_documents(docs_to_add)
#     vector_db.save()
    
#     vector_db= FaissVectorbase.load(embeddings_model, index_filename="hf.index", metadata_filename="hf.pkl")
    
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
    
#     extracted_info,usage = llm_api.call(message=messages[0]) 
#     print(extracted_info)
    api_responses = asyncio.run(parallel_api_calls(llm_api, messages))
    
    extracted_info_dict = process_api_extractions(api_responses)
    
    extracted_info_dict = ensure_all_info_keys_present(extracted_info_dict, sum(info_to_extract_batches, []))
    print(extracted_info_dict)
    
    
    

    
    
        
        
    
    