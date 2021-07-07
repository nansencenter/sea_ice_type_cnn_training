# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:08:35 2021

@author: Alissa Kouraeva
"""

import numpy as np

def form_of_ice(stage):
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
    index_= np.nan
    if stage ==0:
        index_ = 0
        #print('ice_free')

    if stage!=-9:
        if stage in range(81,86):
            #print('Young ice')
            index_=1
        if stage in range(86,94):
            #print('First year ice')
            index_=2
        if stage in range(95,98):
            #print('multiyear ice')
            index_=3
    return index_
    

def one_hot_m1(ct,ca,sa,fa,cb,sb,fb,cc,sc):
    """
    
    Returns the list of one-hot encoded values corresponding to ice types 
    based on concentration and stage of development of thickest, second 
    thickest and thrid thickest ice
    
    Parameters
    ----------
    ct : integer
        Total concentration.
    ca : integer
        Partial concentration of thickest ice.
    sa : integer
        Stage of development of thickest ice.
    fa : integer
        Form of thickest ice.
    cb : integer
        Partial concentration of second thickest ice.
    sb : integer
        Stage of development of second thickest ice.
    fb : integer
        Form of second thickest ice.
    cc : integer
        Partial concentration of third thickest ice.
    sc : integer
        Stage of development of third thickest ice.

    Returns
    -------
    result : list
        List of one-hot encoded values corresponding to ice types.

    """
    
    L=[ct,ca,cb,cc]
    index = np.argmax(L)
    #print(index)
    if index ==0:
        result = [1,0,0,0]
    else :
        #print([sa,sb,sc])
        #print('bibi',[sa,sb,sc][index])
        index2 = form_of_ice([sa,sb,sc][index])
        #print(index2)
        result = [0,0,0,0]
        result[index2]=1
    return result

def one_hot_m2(ct,ca,sa,fa,cb,sb,fb,cc,sc):
    """
    

    Returns the list of one-hot encoded values in terms of concentration 
    corresponding to ice types based on concentration and stage of development
    of thickest, second thickest and thrid thickest ice.
    
    Parameters
    ----------
    ct : integer
        Total concentration.
    ca : integer
        Partial concentration of thickest ice.
    sa : integer
        Stage of development of thickest ice.
    fa : integer
        Form of thickest ice.
    cb : integer
        Partial concentration of second thickest ice.
    sb : integer
        Stage of development of second thickest ice.
    fb : integer
        Form of second thickest ice.
    cc : integer
        Partial concentration of third thickest ice.
    sc : integer
        Stage of development of third thickest ice.

    Returns
    -------
    result : list
        List of one-hot encoded (in terms of concentration) values 
        corresponding to ice types.

    """
    result = [0,0,0,0]
    result[0] = int(ct)/100
    for si, ci in zip([sa,sb,sc], [ca,cb,cc]):
        #print(si)
        index_2 = form_of_ice(si)
        result[index_2] += ci/100
    return result
