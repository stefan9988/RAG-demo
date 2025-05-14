import json
import itertools
from typing import List, Dict, Any

def get_info_to_extract(filename: str = "data/info_to_extract.json") -> List[str]:
    """
    Loads information to be extracted from a JSON file.
    Args:
        filename (str, optional): The path to the JSON file.
            Defaults to "data/info_to_extract.json".

    Returns:
        List[str]: A flat list of strings, where each pair of strings
                   corresponds to a 'Key' and its 'Description' from
                   the input JSON. For example:
                   [key1, description1, key2, description2, ...].    
    """
    with open(filename, 'r', encoding='utf-8') as f:
        info_to_extract_json = json.load(f)
    info_to_extract = [item for d in info_to_extract_json['policy'] for item in (d['Key'], d['Description'])]
    return info_to_extract

def get_unique_dictionaries_by_document_id(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts unique dictionaries from a list based on a 'document id' key,
    keeping the last encountered dictionary for each 'document id'.
    The resulting list of unique dictionaries is then sorted in ascending
    order based on their 'document id'.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries. Each dictionary
            is expected to contain a 'document id' key. If 'document id'
            is missing or None for an item, that item will be ignored.

    Returns:
        List[Dict[str, Any]]: A list of unique dictionaries, sorted by
            'document id' in ascending order. If multiple input dictionaries
            share the same 'document id', only the one that appeared latest
            in the input `data` list is retained.
    """
    unique_dict = {}
    for item in data:
        doc_index = item.get('document id')
        
        if doc_index is not None:
            unique_dict[doc_index] = item
    
    result = list(unique_dict.values())
    result.sort(key=lambda x: x.get('document id', 0))
    return result
    
def create_similar_docs_batches(similar_docs: List[List[Dict[str, Any]]], batch_size: int = 8) -> List[List[Dict[str, Any]]]:
    """
    Processes a list of lists of similar documents (typically search results
    for multiple queries) into de-duplicated and sorted batches.

    The process involves three main steps for the input `similar_docs`:
    1.  Batching: The outer list (representing results from multiple queries)
        is divided into batches of `batch_size`. Each batch will contain
        `batch_size` lists of document results.
    2.  Flattening: Within each of these batches, all document dictionaries
        from the constituent query result lists are combined (flattened)
        into a single list of documents.
    3.  De-duplication and Sorting: The `get_unique_dictionaries_by_document_id`
        function is applied to each flattened batch to remove duplicate documents
        (based on 'document id', keeping the last seen) and sort them.

    Args:
        similar_docs (List[List[Dict[str, Any]]]): A list where each inner list
            contains dictionaries representing documents retrieved for a specific
            query. Each document dictionary is expected to have a 'document id' key.
            Example: [[doc1_q1, doc2_q1], [doc1_q2, doc2_q2, doc3_q2], ...]
        batch_size (int, optional): The number of query result lists to group
            together in the initial batching step. Defaults to 8.

    Returns:
        List[List[Dict[str, Any]]]: A list of batches. Each batch is a list of
            unique document dictionaries (sorted by 'document id') that were
            aggregated from a set of original query results.
    """
    similar_docs_batches = [similar_docs[i:i + batch_size] for i in range(0, len(similar_docs), batch_size)]
    similar_docs_batches = [list(itertools.chain.from_iterable(batch)) for batch in similar_docs_batches]
    similar_docs_batches = [get_unique_dictionaries_by_document_id(batch) for batch in similar_docs_batches]
    return similar_docs_batches