import os
import sys

import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import archive


def read_input_params():
    datapath = sys.argv[1]
    outpath = sys.argv[2]
    nersc = sys.argv[3]  # Leave as empty string '' for ESA noise corrections or as 'nersc_'
                         # for the Nansen center noise correction.
    rm_swath = 0
    distance_threshold = 0
    amsr_labels = [
        "btemp_6.9h",
        "btemp_6.9v",
        "btemp_7.3h",
        "btemp_7.3v",
        "btemp_10.7h",
        "btemp_10.7v",
        "btemp_18.7h",
        "btemp_18.7v",
        "btemp_23.8h",
        "btemp_23.8v",
        "btemp_36.5h",
        "btemp_36.5v",
        "btemp_89.0h",
        "btemp_89.0v",
    ]
    window_size_amsr2 = (14, 14)
    window_size = (window_size_amsr2[0] * 50, window_size_amsr2[1] * 50)
    stride_ams2_size = 14
    stride_sar_size = stride_ams2_size * 50

    sar_names = [nersc + "sar_primary", nersc + "sar_secondary"]
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    archive_ = archive(
        sar_names,
        nersc,
        stride_sar_size,
        stride_ams2_size,
        window_size,
        window_size_amsr2,
        amsr_labels,
        distance_threshold,
        rm_swath,
        outpath,
        datapath,
    )
    return archive_


def main():

    archive_ = read_input_params()
    archive_.get_unprocessed_files()
    for i, filename in enumerate(archive_.files):
        print("Starting %d out of %d unprocessed files" % (i, len(archive_.files)))
        fil = nc.Dataset(os.path.join(archive_.DATAPATH, filename))
        if archive_.check_file_healthiness(fil, filename):
            archive_.read_file_info(fil, filename)
            archive_.calculate_mask(fil)
            archive_.calculate_batches_for_masks()
            archive_.define_util()
            for switch in ["sar","output","amsr2"]:
                archive_.pad_and_batch(fil,switch)
                archive_.calculate_variable_ML(switch)

            # saving section
            archive_.write_scene_files()
            archive_.update_processed_files(i)
        del fil


if __name__ == "__main__":
    main()
