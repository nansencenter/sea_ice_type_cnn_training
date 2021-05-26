import os
from os.path import basename, isdir, join
import argparse
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

from data_generator import DataGeneratorFromMemory
from utility import Configure, type_for_nersc_noise, create_model, common_parser, postprocess_the_args
from archive import Archive

class MemoryBasedConfigure(Configure):
    DataGenerator_ = DataGeneratorFromMemory
    extension = ".nc"
    create_model = create_model
    input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
    output_var_name = 'CT'
    amsr2_var_names = ['btemp_6_9h','btemp_6_9v']

    def apply_model(self):
        """
        Firstly, it sets the folder for reconstructed files. Then by calculting the set of unique
        scene_date of files in input_dir folder, it will the model to them and store the result in
        the folder that has been set at the beginning of function.
        scene_date, for example, could be '20180410T084537'.
        Model is automatically selected by tensorflow as the last checkpoint in the ML training
        calculations.
        """
        self.set_the_folder_of_reconstructed_files()
        scene_dates = {
            f.split("_")[0] for f in os.listdir(self.DATAPATH) if f.endswith(self.extension)
                      }
        for scene_date in scene_dates:
            self.scene_date = scene_date
            # list_of_names is filled with all 'npz' files (in file-based config) that starts with
            # the very scene date or with the name of the 'nc' file (in memory-based config) that
            # starts with the very scene date.
            self.list_of_names = [
                                 join(self.DATAPATH, f)
                                 for f in os.listdir(self.DATAPATH)
                                 if (f.startswith(scene_date))
                                 ]
            self.setup_generator()
            self.create_model()
            self.predict_by_model()
            self.reconstruct_the_image_and_reset_archive_PROP()

    def predict_by_model(self):
        """
        First load the latest model by loading the previously saved weights of the model. Then
        calculate the output of model by using the generator for it.
        """
        latest = tf.train.latest_checkpoint("models")
        self.model.load_weights(latest)
        self.y_pred = self.model.predict(self.training_generator, verbose=0, steps=None,
                            callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    def reconstruct_the_image_and_reset_archive_PROP(self):
        """
        Since the ML model works with patches of image, the final image should be reconstructed
        in order to be understandable as a whole scene by humans. This function first instantiate the
        image with a proper size. Then assemble the patches into the array of full of zeros to make
        the reconstructed image in it. Finally, it saves the result in the '.npz' file.

        This function also resets the archive PROP for memory management purposes.
        """
        self.instantiate_image_with_zeros_and_get_the_patch_locations_of_image()
        # Since 'img_locs' are in patch manner, they need to multiply by window size
        # in order to be in the correct locations in the reconstructed image
        self.img_locs = np.multiply(self.patch_locs, self.WINDOW_SIZE)
        ws0, ws1 = self.WINDOW_SIZE[0], self.WINDOW_SIZE[1]
        for i in range(self.y_pred.shape[0]):
            # assembling every single output of ML network into correct place of image one by one
            self.img[
                    self.img_locs[i][0]:self.img_locs[i][0] + ws0,
                    self.img_locs[i][1]:self.img_locs[i][1] + ws1
                    ] = self.y_pred[i,:,:,0]
        np.savez(
                join(self.reconstruct_path, f"{self.scene_date}_reconstruct.npz"), self.img
                )
        del self.img, self.y_pred
        del self.archive.PROP
        self.archive.PROP = {}

    def set_the_folder_of_reconstructed_files(self):
        """
        With the help of set_path_for_reconstruct function the 'reconstructs_folder' will be created
        at the same level of input directory (for memory-based config) or one level up in foldering
        hierarchy (for file-based config) in order not to put the reconstructed ones and npz files
        in the same folder. This function is only for folder management.
        """
        self.reconstruct_path = join(
                        self.set_path_for_reconstruct(self.DATAPATH), "reconstructs_folder"
                                    )
        if not isdir(self.reconstruct_path):
            os.mkdir(self.reconstruct_path)

    @staticmethod
    def set_path_for_reconstruct(x):
        """helper function for 'set_the_folder_of_reconstructed_files' function."""
        return x

    def calculate_dims(self):
        """ based on the input arguments, the dimensions will be set. """
        # obtaining the shape from the archive
        self.dims_output = self.WINDOW_SIZE
        self.dims_input = self.WINDOW_SIZE
        self.dims_amsr2 = self.WINDOW_SIZE_AMSR2

    def filling_id_list(self):
        """
        'self.list_of_names' has only the name of the '.nc' file with memory based config. Thus,
        it will be used for calculting the patches all of locations of them in memory. Finally, the
        ID list will be filled with the patches locations after memory-based calculations.
        """
        fil = Dataset(*self.list_of_names)
        self.archive.process_dataset(fil, basename(*self.list_of_names))
        self.id_list = self.archive.PROP['_locs']


    def instantiate_image_with_zeros_and_get_the_patch_locations_of_image(self):
        """
        Get the shape of amsr2 array by reading the '.nc' file. reconstructed image is in the size
        of amsr2 image multiplied by the aspect ratio.
        """
        shape_amsr2 = Dataset(*self.list_of_names)['btemp_6.9h'].shape
        self.img = np.zeros(shape=np.multiply(shape_amsr2, self.ASPECT_RATIO))
        self.patch_locs = self.archive.PROP['_locs']

    def set_params(self):
        """
        create 'self.params' with one addition part that holds the information in memory
        """
        super().set_params()
        self.params['prop'] = self.archive.PROP

def read_input_params_for_applying():
    parser = common_parser()
    parser.add_argument('-bs','--batch_size', required=False, type=int,
                        help="batch size for data generator")
    arg = parser.parse_args()
    dict_for_archive_init = postprocess_the_args(arg)
    dict_for_archive_init["apply_instead_of_training"] = True
    dict_for_archive_init["shuffle_on_epoch_end"] = False
    dict_for_archive_init["shuffle_for_training"] = False
    dict_for_archive_init["precentage_of_training"] = 1.
    dict_for_archive_init["batch_size"] = arg.batch_size
    return Archive(**dict_for_archive_init)

def main():

    archive_ = read_input_params_for_applying()

    config_ = MemoryBasedConfigure(archive=archive_)

    config_.apply_model()



if __name__ == "__main__":
    main()
