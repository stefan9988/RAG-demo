import json
import itertools
from typing import List, Dict, Any, Hashable, Iterable

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
    If a document id is encountered again with a different 'query', the new query
    is appended to a list of queries for that document id.
    The resulting list of unique dictionaries is then sorted in ascending
    order based on their 'document id'.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries. Each dictionary
            is expected to contain a 'document id' key and may contain a 'query' key.
            If 'document id' is missing or None for an item, that item will be ignored.

    Returns:
        List[Dict[str, Any]]: A list of unique dictionaries, sorted by
            'document id' in ascending order. For each unique 'document id', the
            dictionary will contain all encountered 'query' values in a 'queries' list
            and will not have a 'query' key.
    """
    unique_dict = {}
    
    for item in data:
        doc_id = item.get('document id')
        key = item.get('key')
        
        if doc_id is not None:
            if doc_id in unique_dict:
                # Document ID already exists
                existing_item = unique_dict[doc_id]
                
                # If the document has a query
                if key is not None:
                    # Ensure there's a 'queries' list
                    if 'keys' not in existing_item:
                        existing_item['keys'] = []
                    
                    # Add the new query to the list if not already present
                    if key not in existing_item['keys']:
                        existing_item['keys'].append(key)
                
                # Update the item with latest data (keeping the queries list)
                for key, value in item.items():
                    if key != 'key':  # Don't copy the query field
                        existing_item[key] = value
            else:
                # New document ID - create a copy and initialize
                new_item = item.copy()
                
                # Initialize keys list if there's a query
                if key is not None:
                    new_item['keys'] = [key]
                    # Remove the individual query key
                    if 'key' in new_item:
                        del new_item['key']
                else:
                    new_item['keys'] = []
                
                unique_dict[doc_id] = new_item
    
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

def create_key_description_pairs(similar_docs: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    key_description_pairs = [similar_docs[i:i + 2] for i in range(0, len(similar_docs), 2)]
    key_description_pairs = [set_shared_query_and_description(pair) for pair in key_description_pairs]
    key_description_pairs = [list(itertools.chain.from_iterable(batch)) for batch in key_description_pairs]
    return key_description_pairs

def set_shared_query_and_description(data_pair):
    """
    Modifies two lists of dictionaries in-place based on their first elements.

    - For each dictionary in the first list (key_list):
        - Sets 'description' to the 'key' value of the first dictionary
          in the second list (description_list).
    - For each dictionary in the second list (description_list):
        - Sets 'description' to the 'key' value of the first dictionary
          in the second list.
        - Sets 'key' to the 'key' value of the first dictionary
          in the first list.

    Args:
        data_pair (tuple or list): A pair of lists, where each list
                                   is expected to contain dictionaries.
                                   data_pair[0] is key_list.
                                   data_pair[1] is description_list.

    Returns:
        list: The modified data_pair (containing the modified key_list
              and description_list).
    """
    
    key_list = data_pair[0]
    description_list = data_pair[1]

    try:
        shared_desc_val = description_list[0]['key']
        shared_query_val_for_desc_list = key_list[0]['key']
    except KeyError as e:
        raise KeyError(f"Required 'key' key missing from first element: {e}")

    for item in key_list:
        item['description'] = shared_desc_val

    for item in description_list:
        item['description'] = shared_desc_val
        item['key'] = shared_query_val_for_desc_list

    # Returning the modified lists is fine if the caller expects it,
    # but be aware of the in-place modification.
    return [key_list, description_list]





def extract_unique_keys_per_batch(
    relevant_docs_batches: Iterable[Iterable[Dict[str, Iterable[Hashable]]]]
) -> List[List[Hashable]]:
    """
    Extracts unique 'keys' from batches of documents.

    For each batch of documents, this function:
    1. Collects the list associated with the 'keys' field from each document.
    2. Flattens these lists of keys for the current batch.
    3. Finds the unique keys within that flattened list for the batch.
    4. Returns a list of these unique key lists, one for each input batch.

    Args:
        relevant_docs_batches: An iterable of batches. 

    Returns:
        A list of lists. Each inner list contains the unique hashable items
        (keys) found across all documents in the corresponding input batch.        
    """
    temp_info_batches = [[doc['keys'] for doc in docs_batch] for docs_batch in relevant_docs_batches]
    processed_batches = [list(set(itertools.chain.from_iterable(batch_of_key_lists)))
        for batch_of_key_lists in temp_info_batches]

    return processed_batches

def join_document_content_per_batch(
    relevant_docs_batches: Iterable[Iterable[Dict[str, str]]],
    separator: str = "\n\n"
) -> List[str]:
    """
    Joins the content of documents within each batch into a single string.

    For each batch of documents, this function:
    1. Extracts the string content associated with 'document content' from each document.
    2. Joins these extracted strings together using the specified `separator`.
    3. Returns a list where each element is the joined string for the corresponding batch.

    Args:
        relevant_docs_batches: An iterable of batches. Each batch is an
            iterable of documents (dictionaries). 
        separator (str, optional): The string used to join the document contents
            within each batch. Defaults to "\n\n".

    Returns:
        A list of strings. Each string is the concatenation of document contents
        from the corresponding input batch, separated by `separator`.        
    """
    doc_content_batches: List[str] = []
    for docs_batch in relevant_docs_batches:
        contents_in_batch = [doc['document content'] for doc in docs_batch]
        joined_content = separator.join(contents_in_batch)
        doc_content_batches.append(joined_content)
    return doc_content_batches