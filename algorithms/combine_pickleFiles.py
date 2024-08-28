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

# Example usage
pickle_file1 = 'pickle1_llm_cache.pickle'
pickle_file2 = 'pickle2_llm_cache.pickle'
combine_pickles(pickle_file1, pickle_file2, 'combined_file.pickle')




