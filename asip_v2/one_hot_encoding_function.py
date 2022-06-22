import numpy as np


def ice_type(stage):
    """
    Gives back the index the concentration or 1 should be on.
    Each index corresponds to a particular ice type
    (0: Young ice; 1: First Year ice; 2: Multi year ice ; 3: Ice free).
    The values on which depends this classification are described ine the
    ASIP-v2 manual.
    Parameters
    ----------
    stage : integer
        stage of development.
    Returns
    -------
    index_ : integer
        index of the list where the value (0/1 or concentration) will be.
    """
    index_= None

    if stage in range (0, 83):
        #print('ice_free')
        index_ = 0
    if stage in range(83, 86):
        #print('Young ice')
        index_=1
    if stage in range(87, 94):
        #print('First year ice')
        index_=2
    if stage in range(95, 98):
        #print('multiyear ice')
        index_=3
    return index_


def one_hot_continous_hugo(vector_param, list_combi):
    """
    Returns the list of one-hot encoded values in terms of concentration
    corresponding to ice types based on concentration and stage of development
    of thickest, second thickest and thrid thickest ice.
    
    Parameters
    ----------
    vector_param : list
        all parameters in a vector.
    list_combi : list
        list of the all work combinations.
    
    Returns
    -------
    result : list
        List of one-hot encoded (in terms of concentration) values
        corresponding to ice types.
    """
    
    result = [0,0,0,0]
    vector_param = vector_param.squeeze()
    for ice in range(3): # in a output there are 3 data for the 3 most present ice
        if vector_param[1+ice*3]==(-9): 
            continue
        if vector_param[2+ice*3]==(-9): 
            continue
        icetype = ice_type(vector_param[2+ice*3])
        result[icetype] += vector_param[1+ice*3]/100
    if max(result) == 0:
         result[0] = 1
    else:
         result[0] = 1-sum(result[1:])
            
    return result


def one_hot_continous_sod_f(vector_param, list_combi):
    """
    Converts the output parameter vector ([ct,ca,sa,fa,...]) 
    into a vector that contains the concentration percentages for the combinations.
    
    Parameters
    ----------
    vector_param : list
        all parameters in a vector.
    list_combi : list
        list of the all work combinations.
    
    Returns
    -------
    result : list
        List of percentage concentrations for each work combination.
    """
    
    result = [0]*len(list_combi)
    vector_param = vector_param.squeeze()
    if vector_param[0] <10 : #open weter
        combi = "0_0"
        index_combi = list_combi.index(combi)
        result[index_combi] = 1
    for ice in range(3): # in a output there are 3 data for the 3 most present ice
        if vector_param[1+ice*3] == (-9): 
            continue
        if vector_param[2+ice*3] == (-9): 
            continue
        if vector_param[3+ice*3] == (-9): 
            continue
        combi = str(int(vector_param[2+ice*3])) + '_' + str(int(vector_param[3+ice*3]))
        index_combi = list_combi.index(combi)
        result[index_combi] += vector_param[1+ice*3]/100

    return result