import os
import sys
import argparse

import netCDF4 as nc
import numpy as np
from skimage.util.shape import view_as_windows

from archive import Archive, SarBatches, OutputBatches, Amsr2Batches
from utility import read_input_params


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
