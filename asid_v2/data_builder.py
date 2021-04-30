import os
import sys
import argparse

import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import Archive, SarBatches, OutputBatches, Amsr2Batches


def read_input_params():
    ASPECT_RATIO = 50
    def type_for_stride_and_window_size(str_):
        if int(str_)%ASPECT_RATIO:
            parser.error(f"Both stride and window size must be dividable to {ASPECT_RATIO}")
        return int(str_)
    def type_for_nersc_noise(str_):
        if not (str_=="" or str_=="nersc_"):
            parser.error("'--noise_method' MUST be '' or 'nersc_'.")
        return str_
    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument(
        'input_dir', type=str, help="Path to directory with input netCDF files")
    parser.add_argument(
        '-o','--output_dir', type=str, required=False,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output"),
        help="Path to directory with output files",)
    parser.add_argument(
        '-n', '--noise_method', required=False, type=type_for_nersc_noise,default="nersc_",
        help="the method that error calculation had been used for error.Leave as empty string '' for"
                    "ESA noise corrections or as 'nersc_' for the Nansen center noise correction.")
    parser.add_argument(
        '-w', '--window_size', required=False, type=type_for_stride_and_window_size,default=700,
        help="window size for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-s', '--stride', required=False, type=type_for_stride_and_window_size,default=700,
        help="stride for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-i', '--inference_mode', action='store_true',
        help="Save all locations of the scene for inference purposes of the scene (not for training).")
    parser.add_argument(
        '-r','--rm_swath', required=False, type=int,default=0,
        help="threshold value for comparison with file.aoi_upperleft_sample to border the calculation")
    parser.add_argument(
        '-d','--distance_threshold', required=False, type=int,default=0,
        help="threshold for distance from land in mask calculation")
    parser.add_argument(
        '-a','--step_resolution_sar', required=False, type=int,default=1,
        help="step for resizing the sar data")
    parser.add_argument(
        '-b','--step_resolution_output', required=False, type=int,default=1,
        help="step for resizing the output variables")
    arg = parser.parse_args()
    window_size_amsr2 = (arg.window_size // ASPECT_RATIO, arg.window_size // ASPECT_RATIO)
    stride_ams2_size = arg.stride // ASPECT_RATIO
    window_size = (arg.window_size, arg.window_size)
    stride_sar_size = arg.window_size
    rm_swath = arg.rm_swath
    distance_threshold = arg.distance_threshold
    datapath = arg.input_dir
    outpath = arg.output_dir
    nersc = arg.noise_method
    step_sar = arg.step_resolution_sar
    step_output = arg.step_resolution_output
    inference_mode = arg.inference_mode
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
        step_sar,
        step_output,
        inference_mode
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
            for cls_ in [SarBatches, OutputBatches, Amsr2Batches]:
                obj = cls_(archive_)
                obj.pad_and_batch(fil)
                archive_.PROP.update(obj.calculate_variable_ML())
                del obj
            del archive_.mask_batches_amsr2, archive_.mask_batches
            # saving section
            archive_.write_scene_files_and_reset_archive_PROP()
            archive_.update_processed_files(i)
        del fil


if __name__ == "__main__":
    main()
