import os
import sys
import argparse

import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import Archive
from utility import read_input_params


def main():

    archive_ = read_input_params()
    archive_.get_unprocessed_files()
    for i, filename in enumerate(archive_.files):
        print("Starting %d out of %d unprocessed files" % (i, len(archive_.files)))
        fil = nc.Dataset(os.path.join(archive_.DATAPATH, filename))
        archive_.calculate_PROP_of_archive(fil, filename)
        # saving section
        archive_.write_scene_files_and_reset_archive_PROP()
        archive_.update_processed_files(i)
        del fil


if __name__ == "__main__":
    main()
