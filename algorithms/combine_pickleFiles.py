import pickle

def combine_pickles(pickle_file1, pickle_file2, output_pickle_file):
    # Load the first pickle file
    with open(pickle_file1, 'rb') as f1:
        data1 = pickle.load(f1)
    
    # Load the second pickle file
    with open(pickle_file2, 'rb') as f2:
        data2 = pickle.load(f2)
    
    # Combine the data (assuming both are dictionaries, or you can modify this as needed)
    if isinstance(data1, dict) and isinstance(data2, dict):
        combined_data = {**data1, **data2}
    else:
        combined_data = [data1, data2]
    
    # Save the combined data to a new pickle file
    with open(output_pickle_file, 'wb') as output_file:
        pickle.dump(combined_data, output_file)
    
    print(f"Combined pickle file saved as {output_pickle_file}")

def update_pickle_data_format(pickle_file1):
    with open(pickle_file1, 'rb') as f1:
        llm_cache = pickle.load(f1)
    new_data = {}
    for key, reward_dict in llm_cache.items():
        data = new_data.setdefault(key, {})
        data["rewards"] = reward_dict

    with open(pickle_file1, 'wb') as f1:
        pickle.dump(new_data, f1)
    return new_data

if __name__=="__main__":
    # Example usage
    pickle_file1 = 'PickleCacheAndBaseModels/Combined/Milk/gpt-4o-mini_llm_cache_1.pickle'
    pickle_file2 = 'PickleCacheAndBaseModels/Combined/Milk/gpt-4o-mini_llm_cache_2.pickle'
    combine_pickles(pickle_file1, pickle_file2, 'PickleCacheAndBaseModels/Combined/Milk/gpt-4o-mini_llm_cache_combined.pickle')
    # update_pickle_data_format('PickleCacheAndBaseModels/Milk/gpt-4o-mini_llm_cache.pickle')
    # update_pickle_data_format('PickleCacheAndBaseModels/Drive/gpt-4o-mini_llm_cache.pickle')