def get_first_and_second(path):
    """
    Read and construct the label mapping for two-level classification.
    
    Args:
        path (str): The file path containing classification label data.
        
    Returns:
        tuple: Contains four dictionaries:
            - first_id2name: Mapping from first-level category id to name.
            - first_name2id: Mapping from first-level category name to id.
            - second_id2name: Mapping from second-level category id to name.
            - second_name2id: Mapping from second-level category name to id.
    """
    first_id2name = {}    # Mapping from first-level category id to name
    first_name2id = {}    # Mapping from first-level category name to id
    second_id2name = {}   # Mapping from second-level category id to name
    second_name2id = {}   # Mapping from second-level category name to id
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            first_id, first_name, second_id, second_name = line.strip().split(',')
            
            first_id = int(first_id)
            first_id2name[first_id] = first_name
            first_name2id[first_name] = first_id
            
            second_id = int(second_id)
            second_id2name[second_id] = second_name
            second_name2id[second_name] = second_id
            
    except Exception as e:
        print(f"Error reading classification label file: {e}")
        raise
        
    return first_id2name, first_name2id, second_id2name, second_name2id