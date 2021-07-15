import os
import sys
import argparse
import argparse
import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import Archive
from utility import type_for_nersc_noise, common_parser, postprocess_the_args

def read_input_params_for_building():
    """
    read the input data based on the command line arguments and return an instance of archive class
    """
    parser = common_parser()
    arg = parser.parse_args()
    dict_for_archive_init = postprocess_the_args(arg)
    return Archive(**dict_for_archive_init)

def main():

    archive_ = read_input_params_for_building()
    archive_.get_unprocessed_files()
    for i, filename in enumerate(archive_.files):
        print("Starting %d out of %d unprocessed files" % (i, len(archive_.files)))
        fil = nc.Dataset(os.path.join(archive_.DATAPATH, filename))
        archive_.process_dataset(fil, filename)
        # saving section
        archive_.write_batches()
        archive_.update_processed_files(i)
        del fil


if __name__ == "__main__":
    main()
