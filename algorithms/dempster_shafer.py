import math
import numpy as np
import pandas as pd
    
from itertools import combinations
from collections import defaultdict

EPSILON = 1e-12


def H(x):
    '''shannon entropy'''
    return -np.sum(x*np.log2(x))

def belief_to_reward(belief_dict, actionsets):
    array = [value for value in belief_dict.values()]
    transposed_array = np.array(array)
    belief_matrix = transposed_array + EPSILON*(transposed_array==0) # add small epsilon to null values
    
    #Step 1-1 Construct the distance measure matrix DMM = BJS(ij) as follows:
    # Initialize a 4x5 array with default values (e.g., zeros)
    rows, num_clusters = 4, len(belief_dict) #rows are number of actions. cols are number of model clusters (sensors)
    DMM = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(num_clusters):
            DMM[i,j] = H(0.5*(belief_matrix[i,:]+belief_matrix[j, :])) -0.5*(H(belief_matrix[i, :]) + H(belief_matrix[j, :]))

    # print(DMM)

    #Step 1-2 Obtain the average evidence distance BJS_i of the evidence m_i follows:
    # Get the sum of each row
    BJS = sum_rows(DMM, num_clusters)
    # print(BJS)

    #Step 1-3 Calculate the support degree of the evidence mi as below:
    # Divide each element by 1 
    SUP = 1/BJS
    # print(SUP)

    #Step 1-4 : Compute the credibility degree of the evidence mi as follows:
    CRD = SUP/np.sum(SUP)
    # print(CRD)

    #Step 2-1: Measure the belief entropy of the evidence mi as below:
    ED = compute_belief_entropy(belief_matrix, actionsets)
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
    
    # frame = list(belief_dict.keys())
    frame = actionsets
    frameset = frozenset().union(*frame)

    # Create a list of 5 identical BBAs ??WHY??
    # v =  [0.5316,0.1472,0.0521,0.2692]
    # bbas = [create_bba(WAE_m, frame) for _ in range(k)]
    bbas = [{k: v for k,v in zip(frame, WAE_m)}]*num_clusters
    # print(bbas)
    # for i, bba in enumerate(bbas, start=1):
    #     print(f"BBA {i}:")
    #     for key, value in sorted(bba.items()):
    #         print(f"  {set(key)}: {value:.4f}")
    #     print()


    # Note the number of times it is combined is based on K
    combined_bba = bbas[0]
    for i in range(num_clusters-1):
        combined_bba = dempster_combination_rule(combined_bba, bbas[i+1], frameset)
    return combined_bba
    

# Get BBAs from WAE_m and compute number of times it needs to be combined. If K = number of sensors. In our case it is 5. So, it will be combined 4 times. 

def dempster_combination_rule(bba1, bba2, frame):
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



def compute_belief_entropy(array, actionsets):
    # a1 = 0.60;a2=0.10;a3=1e-12;a4=0.30 # nope
    cardinality = 1 #number of element in a set. {A} = 1 {A,C} = 2
    num_sensors, num_actions = array.shape
    assert len(actionsets)==num_actions
    ED = np.zeros(num_sensors)    
    for index, col in enumerate(array):
        ed = 0
        for col_index, element in enumerate(col):
            cardinality = len(actionsets[col_index])
            ed += element*np.log2(element/((2**cardinality) - 1))
        ED[index] = -ed
    return ED

def compute_information_volume(array):
    for i in range(len(array)):
        array[i] = math.exp(array[i])
    return array


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
    transpose_matrix = np.array([
        [0.41, 0.29, 0.3, 0],
        [0, 0.9, 0.1, 0],
        [0.58, 0.07, 0, 0.35],
        [0.55, 0.1, 0, 0.35],
        [0.6, 0.1, 0, 0.3]
    ])
    # transpose_matrix[0][0]=0.41;transpose_matrix[0][1]=0.29;transpose_matrix[0][2]=0.30;transpose_matrix[0][3]=1e-12
    # transpose_matrix[1][0]=1e-12;transpose_matrix[1][1]=0.90;transpose_matrix[1][2]=0.10;transpose_matrix[1][3]=1e-12
    # transpose_matrix[2][0]=0.58;transpose_matrix[2][1]=0.07;transpose_matrix[2][2]=1e-12;transpose_matrix[2][3]=0.35
    # transpose_matrix[3][0]=0.55;transpose_matrix[3][1]=0.10;transpose_matrix[3][2]=1e-12;transpose_matrix[3][3]=0.35
    # transpose_matrix[4][0]=0.60;transpose_matrix[4][1]=0.10;transpose_matrix[4][2]=1e-12;transpose_matrix[4][3]=0.30

    actionsets = [frozenset({'A'}), frozenset({'B'}),frozenset({'C'}),frozenset({'A','C'})]
    # WAE_ref = [0.5315501408507114, 0.14715770640831496, 0.052076400657720116, 0.26921575208428306]
    belief_dict = {mor: actions for mor, actions in enumerate(transpose_matrix)}
    
    rews = belief_to_reward(belief_dict, actionsets)

    rews_reference = eval("{frozenset({'A'}): 0.9894584020940587, frozenset({'B'}): 0.00020828314107542985, frozenset({'C'}): 0.006065147778246115, frozenset({'A', 'C'}): 0.00426816699793185}")

    # print(rews)
    keylist = rews.keys()
    result = np.array([rews[key] for key in keylist])
    reference = np.array([rews_reference[key] for key in keylist])
    assert np.allclose(result, reference)