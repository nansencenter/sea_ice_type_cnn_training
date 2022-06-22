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

    if stage in range(0,83):
        index_ = 0
        #print('ice_free')
    if stage in range(83,86):
        #print('Young ice')
        index_ = 1
    if stage in range(87,95):
        #print('First year ice')
        index_ = 2
    if stage in range(95,98):
        #print('multiyear ice')
        index_ = 3
    return index_


def one_hot_binary(ct,ca,sa,fa,cb,sb,fb,cc,sc,fc, min_ct=10):

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
    fc : integer
        Form of third thickest ice.
    Returns
    -------
    result : list
        List of one-hot encoded values corresponding to ice types.
    """

    #cabc = [ca,cb,cc]
    result = [0,0,0,0]
    if ct < min_ct:
        return [1,0,0,0] # open water

    f = [0,0,0] # fractions
    for ci,si in zip([ca,cb,cc], [sa,sb,sc]):
        icetype = ice_type(si)
        if ci != -9 and icetype is not None:
           f[icetype-1] += ci
    if max(f) == 0:
        icetype = ice_type(sa)
    else:
        icetype = np.argmax(f)+1
    if icetype is not None:
        result[icetype] = 1

    return result


def one_hot_continous(ct,ca,sa,fa,cb,sb,fb,cc,sc,fc):
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
    fc : integer
        Form of third thickest ice.
    Returns
    -------
    result : list
        List of one-hot encoded (in terms of concentration) values
        corresponding to ice types.
    """
    result = [0,0,0,0]
    for si, ci in zip([sa,sb,sc], [ca,cb,cc]):
        #print(si)
        icetype = ice_type(si)
        if ci != -9 and icetype is not None:
           result[icetype] += ci/100
    if max(result) == 0:
        result[0] = 1-(ct/100)
        icetype = ice_type(sa)
        if icetype is not None  :
            result[icetype] = ct/100
    else:
        result[0] = 1-sum(result[1:])

    return result
