import os
import sys
import argparse

import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import Archive


def read_input_params():
    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument(
        '-in', '--input_source_path', required=True, type=str,
        help="Absolute path for input folder that contains all files")
    parser.add_argument(
        '-out', '--output_destination_path', required=True, type=str,
        help="Absolute path for output folder that storage of all files after processing")
    parser.add_argument(
        '-ne', '--nersc_error_flag', required=True, type=str,
        help="the method that error calculation had been used for error")
    parser.add_argument(
        '-ws', '--window_size', required=True, type=int,
        help="window_size for batching calculation")
    parser.add_argument(
        '-s', '--stride', required=True, type=int,
        help="stride for batching calculation")
    parser.add_argument(
        '--rm_swath', required=True, type=int,
        help="rm_swath")
    parser.add_argument(
        '-dt','--distance_threshold', required=True, type=int,
        help="threshold for distance from land in mask calculation")
    arg = parser.parse_args()
    window_size_amsr2 = (arg.window_size, arg.window_size)
    stride_ams2_size = arg.stride
    rm_swath = arg.rm_swath
    distance_threshold = arg.distance_threshold
    datapath = arg.input_source_path
    outpath = arg.output_destination_path
    nersc = arg.nersc_error_flag  # Leave as empty string '' for ESA noise corrections
                                  # or as 'nersc_' for the Nansen center noise correction.
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
    window_size = (window_size_amsr2[0] * 50, window_size_amsr2[1] * 50)
    stride_sar_size = stride_ams2_size * 50

    sar_names = [nersc + "sar_primary", nersc + "sar_secondary"]
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    archive_ = Archive(
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
            archive_.read_icechart_coding(fil, filename)
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
