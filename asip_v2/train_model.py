import datetime
import os
from os.path import basename, dirname, join

import numpy as np
import tensorflow as tf

from data_generator import DataGeneratorFrom_npz_File
from utility import Configure, read_input_params


class FileBasedConfigure(Configure):
    DataGenerator_ = DataGeneratorFrom_npz_File
    extension = ".npz"

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
                       # TODO: uncomment this line to able callbacks
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




def main():

    archive_ = read_input_params()

    config_ = FileBasedConfigure(archive=archive_)

    config_.train_model()


if __name__ == "__main__":
    main()
