import argparse
import datetime
import os
from os.path import basename, dirname, join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from archive import Archive
from data_generator import DataGeneratorFrom_npz_File
from utility import Configure, create_model, between_zero_and_one_float_type


class FileBasedConfigure(Configure):
    DataGenerator_ = DataGeneratorFrom_npz_File
    extension = ".npz"
    create_model = create_model
    input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
    output_var_name = 'CT'
    amsr2_var_names = ['btemp_6_9h','btemp_6_9v']

    def train_model(self):
        """
        This function is called for training the model. Any training-related configuration of
        Tensorflow can be set here. It saves the model during the training in 'models' folder.
        The names of the models contains the time and date of execution of code as well as the epoch
        that has been reached. Based on Tensorflow abilities, the last one is automatically pointed
        in checkpoint file which can be used afterwards for applying (inference) purposes.
        Also, at the end of training process, it save the final model in 'final_model' folder.
        """
        path_ = self.OUTPATH
        self.list_of_names = [join(path_, f) for f in os.listdir(path_) if (f.endswith(self.extension))]
        self.setup_generator()
        self.create_model()
        self.model.summary()
        # appropriate folder for logging with tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # callback for tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # callback for saving checkpoints
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=join("models",datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+"{epoch:04d}"),
            verbose=1,
            save_weights_only=True,
            save_freq=6 * self.params["batch_size"])
        # Train the model
        self.model.fit(self.training_generator,
                       validation_data=self.validation_generator,
                       use_multiprocessing=True,
                       workers=4,
                       epochs=4,
                       # uncomment this line to enable the callbacks
                       #callbacks=[tensorboard_callback, cp_callback],
                      )
        self.model.save("final_model")

    @staticmethod
    def set_path_for_reconstruct(x):
        """helper function for 'set_the_folder_of_reconstructed_files' function."""
        return dirname(x)

    def calculate_dims(self):
        """
        In file based configure, This function reads the dimensions of data by opening one of 'npz'
        files and get the shape of the variable inside of it.
        """
        # obtaining the shape from the first sample of data
        self.dims_input = np.load(self.id_list[0]).get(self.input_var_names[0]).shape
        self.dims_output = np.load(self.id_list[0]).get(self.output_var_name).shape
        self.dims_amsr2 = np.load(self.id_list[0]).get(self.amsr2_var_names[0]).shape

    def filling_id_list(self):
        """
        In file based configure, id list will be filled with those files that are between two
        specific input dates. Regardless of year of those dates, the date of file is compared to the
        first of jan of year of the same year. This is done for training with data that belongs to a
        specific time period during the year (for example, only for winter times).
        """
        self.id_list = []
        for x in self.list_of_names:
            datetime_ = datetime.datetime.strptime(basename(x).split("_")[0], '%Y%m%dT%H%M%S')
            first_of_jan_of_year = datetime.datetime(datetime_.year, 1, 1)
            if self.BEGINNING_DAY_OF_YEAR <= (datetime_ - first_of_jan_of_year).days <= self.ENDING_DAY_OF_YEAR:
                self.id_list.append(x)

    def instantiate_image_with_zeros_and_get_the_patch_locations_of_image(self):
        """
        'npz' files contain the location of the patch at the end of their name. Based on the maximum
        location of patches in each direction and the window size, the size of reconstructed image
        is determined. Base on patching (that has been done beforehand) this size is multiplication
        of two mentioned numbers (the window size and maximum number of patches in each direction).
        Since the counting is started from zero in python, the maximum number should be incremented
        by one.
        """
        #each locations based on each file name
        self.patch_locs = [(x.split("-")[-1].split(".")[0]).split("_") for x in self.list_of_names]
        # convert them into integer
        self.patch_locs = [(int(x[0]), int(x[1])) for x in self.patch_locs]
        self.img = np.zeros(shape=np.multiply(
                                        self.WINDOW_SIZE,
                                        (max(self.patch_locs)[0] + 1, max(self.patch_locs)[1] + 1)
                                             ))

def read_input_params_for_training():
    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument(
        '-o','--output_dir', type=str, required=True,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output"),
        help="Path to directory with output files (npz files)",)
    parser.add_argument(
        '-see', '--shuffle_on_epoch_end', action='store_true',
        help="Shuffle the training subset of IDs at the end of every epoch during the training.")
    parser.add_argument(
        '-sft', '--shuffle_for_training', action='store_true',
        help="Shuffle the list of IDs before dividing it into two 'training' and 'validation' subsets.")
    parser.add_argument(
        '-bd','--beginning_day_of_year', required=False, type=int, default=0,
        help="min threshold value for comparison with scenedate of files for considering a limited "
             "subset of files based on their counts from the first of january of the same year.")
    parser.add_argument(
        '-ed','--ending_day_of_year', required=False, type=int, default=365,
        help="max threshold value for comparison with scenedate of files for considering a limited "
             "subset of files based on their counts from the first of january of the same year.")
    parser.add_argument(
        '-bs','--batch_size', required=False, type=int,
        help="batch size for data generator")
    parser.add_argument(
        '-p','--precentage_of_training', required=True, type=between_zero_and_one_float_type,
        help="percentage of IDs that should be considered as training data (between 0,1). "
             "'1-precentage_of_training' fraction of data is considered as validation data.")
    arg = parser.parse_args()
    return Archive(
                   outpath=arg.output_dir,
                   shuffle_on_epoch_end=arg.shuffle_on_epoch_end,
                   shuffle_for_training=arg.shuffle_for_training,
                   precentage_of_training=arg.precentage_of_training,
                   beginning_day_of_year=arg.beginning_day_of_year,
                   ending_day_of_year=arg.ending_day_of_year,
                   batch_size=arg.batch_size,
                   apply_instead_of_training=False
                  )

def main():
    archive_ = read_input_params_for_training()

    config_ = FileBasedConfigure(archive=archive_)

    config_.train_model()


if __name__ == "__main__":
    main()
