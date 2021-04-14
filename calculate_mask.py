import numpy as np

def get_the_mask_of_sar_data(sar_names,fil,distance_threshold):
    for str_ in sar_names+['polygon_icechart']:
        mask = np.ma.getmaskarray(fil[str_][:])
        mask = np.ma.mask_or(mask, mask)
        # not only mask itself, but also being finite is import for the data. Thus, another
        # mask also should consider and apply with 'mask_or' of numpy
        mask_isfinite = np.ma.getmaskarray(~np.isfinite(fil[str_]))
        mask = np.ma.mask_or(mask, mask_isfinite)
    # ground data is also masked
    mask_sar_size = np.ma.mask_or(mask, np.ma.getdata(fil['distance_map']) <= distance_threshold)
    return mask_sar_size

def get_the_mask_of_amsr2_data(amsr_labels, fil):
    for amsr_label in amsr_labels:
        mask_amsr = np.ma.getmaskarray(fil[amsr_label])
        mask_amsr = np.ma.mask_or(mask_amsr, mask_amsr)
        mask_isfinite = np.ma.getmaskarray(~np.isfinite(fil[amsr_label]))
        mask_amsr = np.ma.mask_or(mask_amsr, mask_isfinite, shrink=False)
    shape_mask_amsr_0, shape_mask_amsr_1 = mask_amsr.shape[0], mask_amsr.shape[1]
    # enlarging the mask of amsr2 data to be in the size of mask sar data
    mask_amsr = np.repeat(mask_amsr, 50, axis=0)
    mask_amsr = np.repeat(mask_amsr, 50, axis=1)
    return mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1


def padding(mask_amsr, mask_sar_size):
    # the difference between 'amsr repeated mask' and sar size mask must be padded in order to
    # centralize the scene for both sizes and having the same shape of masks
    pad_width = mask_amsr.shape[1]-mask_sar_size.shape[1]
    pad_width_west = pad_width // 2
    if (pad_width % 2) == 0:
        pad_width_east = pad_width // 2
    else:
        pad_width_east = (pad_width // 2) + 1
    pad_hight = mask_amsr.shape[0]-mask_sar_size.shape[0]
    pad_hight_up = pad_hight // 2
    if (pad_hight % 2) == 0:
        pad_hight_down = pad_hight // 2
    else:
        pad_hight_down = (pad_hight // 2) + 1
    mask_sar_size = np.pad(mask_sar_size, ((pad_hight_up, pad_hight_down),
                                           (pad_width_west, pad_width_east)),
                                           'constant', constant_values=(True, True))
    pads = (pad_hight_up, pad_hight_down, pad_width_west, pad_width_east)
    return mask_sar_size, pads

def downsample_mask_for_amsr2(final_ful_mask, shape_mask_amsr_0, shape_mask_amsr_1):
    ##4. final mask must also be available in the amsr2 size, so based on each patch of true/false
    # values inside it, it should be only one single value of true/false in that location.
    final_ful_mask_row_splitted = np.split(final_ful_mask, shape_mask_amsr_0)
    final_mask_with_amsr2_size = np.empty([shape_mask_amsr_0, shape_mask_amsr_1])
    # this loop downsize the final mask to amsr2 size with using squared patches of it
    for ii, some_rows_after_splitting in enumerate(final_ful_mask_row_splitted):
        for jj, square_shape_patch in enumerate(
                                           np.hsplit(some_rows_after_splitting, shape_mask_amsr_1)):
            # below line finds out one single value of true or false
            # based on one patch of true/false values
            unique_value = np.unique(square_shape_patch)
            if unique_value.size == 2:
                # in the case of having both False and True values inside a batch, True must
                # be selected to indicate the very batch is masked batch.
                unique_value = True
            final_mask_with_amsr2_size[ii, jj] = bool(unique_value)
    final_mask_with_amsr2_size = final_mask_with_amsr2_size.astype(bool)
    return final_mask_with_amsr2_size

def calculate_mask(fil, amsr_labels, sar_names, distance_threshold):
    """
    This function has four main calculation sections for calculating the mask:
    1. find out the mask of sar size data. This mask is found based on the mask of all data in file
    with sar size. combination of all masks of sar data is done with "np.ma.mask_or".

    2. find out the mask of amsr2 data. This mask is calculate by combining all the masks of amsr2
    data in the file and the repeat element by element in order to be at the size of sar mask.

    3. final mask is calculated with "np.ma.mask_or" between the sar mask and the amsr2 mask.

    4. final mask should also be available for amsr2 data. Thus, it is downsized by its unique values
    in each patches of it.
    ====
    inputs:
    fil: netCDF4 file object
    amsr_labels: list of names of amsr2 labels in the file
    sar_names: list of names of sar labels in the file
    distance_threshold: integer indicating the threshold for considering the mask based on distance
    to land values.
    ====
    outputs:
    final_ful_mask:
    final_mask_with_amsr2_size: final_ful_mask with amsr2 shape
    pads: used pads for making two images size (coming from sar and amsr2) be at the same number of pixels
    """
    #1. get the mask of sar data
    mask_sar_size = get_the_mask_of_sar_data(sar_names,fil,distance_threshold)
    ##2. get the mask of amsr2 data
    mask_amsr,shape_mask_amsr_0, shape_mask_amsr_1 = get_the_mask_of_amsr2_data(amsr_labels, fil)

    mask_sar_size, pads = padding(mask_amsr, mask_sar_size)
    ##3. final mask based on two masks
    final_ful_mask = np.ma.mask_or(mask_sar_size, mask_amsr)  # combination of masks

    final_mask_with_amsr2_size = downsample_mask_for_amsr2(final_ful_mask, shape_mask_amsr_0, shape_mask_amsr_1)

    return (final_ful_mask, final_mask_with_amsr2_size, pads)
