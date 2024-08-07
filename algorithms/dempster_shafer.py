import math
import numpy as np
import pandas as pd
    
from itertools import combinations
from collections import defaultdict

def replace_zeros_with_small_value(matrix, value=1e-12):
    """
    Replace all zero elements in a 2D matrix with a specified small value.

    Args:
    matrix (list of lists): The 2D array in which to replace zeros.
    value (float): The value to replace zeros with (default is 1e-12).

    Returns:
    list of lists: The matrix with zeros replaced by the specified value.
    """
    # Iterate over each row and each element in the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i, j] == 0:
                matrix[i, j] = value

    return matrix

def H(x):
    '''shannon entropy'''
    return -np.sum(x*np.log2(x))

def belief_to_reward(belief_dict):
    array = [list(map(float, value.split('_'))) for value in belief_dict.values()]
    transposed_array = np.array(array)
    belief_matrix = replace_zeros_with_small_value(transposed_array)
    
    #Step 1-1 Construct the distance measure matrix DMM = BJS(ij) as follows:
    # Initialize a 4x5 array with default values (e.g., zeros)
    rows, cols = 4, 5 #rows are number of actions. cols are number of model clusters (sensors)
    DMM = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(cols):
            DMM[i,j] = H(0.5*(belief_matrix[i,:]+belief_matrix[j, :])) -0.5*(H(belief_matrix[i, :]) + H(belief_matrix[j, :]))

    # print(DMM)

    #Step 1-2 Obtain the average evidence distance BJS_i of the evidence m_i follows:
    # Get the sum of each row
    BJS = sum_rows(DMM,cols)
    # print(BJS)

    #Step 1-3 Calculate the support degree of the evidence mi as below:
    # Divide each element by 1 
    SUP = 1/BJS
    # print(SUP)

    #Step 1-4 : Compute the credibility degree of the evidence mi as follows:
    CRD = SUP/np.sum(SUP)
    # print(CRD)

    #Step 2-1: Measure the belief entropy of the evidence mi as below:
    ED = compute_belief_entropy(belief_matrix)
    # print(ED)

    #Step 2-2: Measure the information volume of the evidence mi as below
    IV = np.exp(ED)
    # print(IV)

    #Step 2-3: Normalise the information volume of the evidence mi as follows:
    IV_Norm = IV/IV.sum()
    # print(IV_Norm)

    assert CRD.shape==IV_Norm.shape
    #Step 3-1: Adjust the credibility degree of the evidence mi based on the information volume of the evidence as below:
    ACrd = CRD*IV_Norm
    # print(ACrd)

    #Step 3-2: Normalise the adjusted credibility degree of the evidence mi as below:
    ACrd_Norm = ACrd/ACrd.sum()
    # print(ACrd_Norm)

    #Step 3-3: Compute the weighted average evidence as follows:
    WAE_m = np.matmul(ACrd_Norm, belief_matrix)
    # print(WAE_m)
    
    frame = list(belief_dict.keys())
    k = cols # TODO: simplify
    # Create a list of 5 identical BBAs ??WHY??
    # v =  [0.5316,0.1472,0.0521,0.2692]
    bbas = [create_bba(WAE_m, frame) for _ in range(k)]
    # print(bbas)
    # for i, bba in enumerate(bbas, start=1):
    #     print(f"BBA {i}:")
    #     for key, value in sorted(bba.items()):
    #         print(f"  {set(key)}: {value:.4f}")
    #     print()


    # Note the number of times it is combined is based on K
    combined_bba = bbas[0]
    for i in range(k-1):
        combined_bba = dempster_combination_rule(combined_bba, bbas[i+1])
    return combined_bba
    

# Get BBAs from WAE_m and compute number of times it needs to be combined. If K = number of sensors. In our case it is 5. So, it will be combined 4 times. 
def create_bba(values, frame):
    return {
        frozenset([frame[0]]): values[0],
        frozenset([frame[1]]): values[1],
        frozenset([frame[2]]): values[2],
        frozenset([frame[3]]): values[3],
    }

def dempster_combination_rule(bba1, bba2):
    frame = ['A', 'B', 'C',  'D']
    frame_subsets = [frozenset(subset) for i in range(len(frame)+1) for subset in combinations(frame, i)]
    
    combined_bba = defaultdict(float)
    total_conflict = 0.0
    
    for subset1 in bba1:
        for subset2 in bba2:
            intersection = subset1 & subset2
            if intersection:
                combined_bba[intersection] += bba1[subset1] * bba2[subset2]
            else:
                total_conflict += bba1[subset1] * bba2[subset2]
    
    if (1 - total_conflict) > 0:
        combined_bba = {key: value / (1 - total_conflict) for key, value in combined_bba.items()}
    else:
        combined_bba = {key: 0.0 for key in combined_bba}
    
    return combined_bba


def sum_rows(matrix,cols):
    return matrix.sum(axis=1)/(cols-1)


def average_elements_by_sum(array):
    total_sum = 0
    for i in array:
        total_sum = total_sum + i
    new_array = [element / total_sum for element in array]
    return new_array

def compute_belief_entropy(array):
    # a1 = 0.60;a2=0.10;a3=1e-12;a4=0.30 # nope
    cardinality = 1 #number of element in a set. {A} = 1 {A,C} = 2
    num_columns = len(array[0])
    num_sensors, num_actions = array.shape
    ED = np.zeros(num_sensors)    
    for index, col in enumerate(array):
        ed = 0
        for col_index, element in enumerate(col):
            cardinality = 1
            if col_index == 3:
                cardinality = 2
            ed += element*np.log2(element/((2**cardinality) - 1))
        ED[index] = -ed
    return ED

def compute_information_volume(array):
    for i in range(len(array)):
        array[i] = math.exp(array[i])
    return array

def normalize_information_volume(array):
    total_sum = 0
    for element in array:
        total_sum += element
    # Normalize each element by dividing it by the total sum
    normalized_array = [x / total_sum for x in array]
    return normalized_array
        
def adjust_credibility_degree(crd,iv_norm):
    result = [crd[i] * iv_norm[i] for i in range(len(crd))]
    return result

def normalize_credibility_degree(array):
    total_sum = 0
    for element in array:
        total_sum += element
    # Normalize each element by dividing it by the total sum
    normalized_array = [x / total_sum for x in array]
    return normalized_array

def weighted_average_evidence(ACrd_Norm,belief_matrix): 
    array = []   
    for j in range(len(belief_matrix)):
        result = 0
        for i in range(len(ACrd_Norm)):
            result = result + (ACrd_Norm[i] * belief_matrix[j][i])
            print
        array.append(result)
    return array



if __name__ == "__main__":
    # Some tests to make sure code still works properly
    ## Results are based on the paper from https://doi.org/10.1016/j.inffus.2018.04.003
    import numpy as np
    rows, cols = 5, 4
    transpose_matrix = np.zeros((rows, cols))
    transpose_matrix[0][0]=0.41;transpose_matrix[0][1]=0.29;transpose_matrix[0][2]=0.30;transpose_matrix[0][3]=1e-12
    transpose_matrix[1][0]=1e-12;transpose_matrix[1][1]=0.90;transpose_matrix[1][2]=0.10;transpose_matrix[1][3]=1e-12
    transpose_matrix[2][0]=0.58;transpose_matrix[2][1]=0.07;transpose_matrix[2][2]=1e-12;transpose_matrix[2][3]=0.35
    transpose_matrix[3][0]=0.55;transpose_matrix[3][1]=0.10;transpose_matrix[3][2]=1e-12;transpose_matrix[3][3]=0.35
    transpose_matrix[4][0]=0.60;transpose_matrix[4][1]=0.10;transpose_matrix[4][2]=1e-12;transpose_matrix[4][3]=0.30

    # WAE_ref = [0.5315501408507114, 0.14715770640831496, 0.052076400657720116, 0.26921575208428306]
    belief_dict = {mor: "_".join([str(x) for x in actions]) for mor, actions in enumerate(transpose_matrix)}
    
    rews = belief_to_reward(belief_dict)

    rews_reference = eval("{frozenset({'A'}): 0.9662198491114329, frozenset({'B'}): 0.001571338743504884, frozenset({'C'}): 8.720865036021161e-06, frozenset({'D'}): 0.03220009134186432}")

    keylist = rews.keys()
    result = np.array([rews[key] for key in keylist])
    reference = np.array([rews_reference[key] for key in keylist])
    assert np.allclose(result, reference)