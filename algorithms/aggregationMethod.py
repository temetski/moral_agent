import math
import numpy as np
import pandas as pd

def aggregate_belief_to_reward(belief_dict,method='weight_average'):
    array = [value for value in belief_dict.values()]
    if method=='weight_average':
        reward = np.mean(np.transpose(array[:5]), axis=1)
    elif method=='arg_max':
        reward = np.max(np.transpose(array[:5]), axis=1)
    elif method=='voting':
        reward = voting_combination_rule(belief_dict)
    return reward

    
def voting_combination_rule(belief_dict):
    array = [value for value in belief_dict.values()]
    array = np.transpose(array[:5])
    results = np.zeros_like(array)
    max_indices = np.argmax(array, axis=0)
    for col_index in range(array.shape[1]):
    # Set the corresponding row to 1 for the max value index
        results[max_indices[col_index], col_index] = 1
    row_sums = np.sum(results, axis=1)
    # Final result: Set 1 for the action with the maximum count and 0 for others
    final_result = np.zeros(row_sums.shape, dtype=int)
    max_index = np.argmax(row_sums)  # Find the index of the maximum value in row_sums
    final_result[max_index] = 1  # Set that index to 1
    return final_result