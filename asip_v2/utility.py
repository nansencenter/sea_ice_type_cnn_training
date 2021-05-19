import argparse
import datetime
import os
import random
from os.path import basename, dirname, isdir, join

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from archive import Archive
from data_generator import DataGeneratorFrom_npz_File, DataGeneratorFromMemory


def read_input_params():
    """
    read the input data based on the command line arguments and return an instance of archive class
    """
    def type_for_nersc_noise(str_):
        if not (str_=="" or str_=="nersc_"):
            parser.error("'--noise_method' MUST be '' or 'nersc_'.")
        return str_
    def between_zero_and_one_float_type(arg):
        """ Type function for argparse - a float within some predefined bounds """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f <= 0. or f > 1.:
            raise argparse.ArgumentTypeError("Argument must be =< " + str(1.) + " and > " + str(0.))
        return f

    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument(
        'input_dir', type=str, help="Path to directory with input netCDF files")
    parser.add_argument(
        '-o','--output_dir', type=str, required=True,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output"),
        help="Path to directory with output files (npz files)",)
    parser.add_argument(
        '-n', '--noise_method', required=False, type=type_for_nersc_noise,default="nersc_",
        help="the method that error calculation had been used for error.Leave as empty string '' for"
                    "ESA noise corrections or as 'nersc_' for the Nansen center noise correction.")
    parser.add_argument(
        '-w', '--window_size', required=False, type=int,default=700,
        help="window size for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-s', '--stride', required=False, type=int,default=700,
        help="stride for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-r', '--aspect_ratio', required=True, type=int,
        help="The ration between the cell size of primary and secondary input of ML model. stride"
        " and window_size must be dividable to it.")
    parser.add_argument(
        '-i', '--apply_instead_of_training', action='store_true',
        help="Consider all locations of the scene for inference purposes of the scene (not for training).")
    parser.add_argument(
        '-see', '--shuffle_on_epoch_end', action='store_true',
        help="Shuffle the training subset of IDs at the end of every epoch during the training.")
    parser.add_argument(
        '-sft', '--shuffle_for_training', action='store_true',
        help="Shuffle the list of IDs before dividing it into two 'training' and 'validation' subsets.")
    parser.add_argument(
        '-m', '--memory_mode', action='store_true',
        help="use memory instead of npz files for the input of inference of the scene (not for training).")
    parser.add_argument(
        '-swa','--rm_swath', required=False, type=int,default=0,
        help="threshold value for comparison with file.aoi_upperleft_sample to border the calculation")
    parser.add_argument(
        '-bd','--beginning_day_of_year', required=False, type=int, default=0,
        help="min threshold value for comparison with scenedate of files for considering a limited "
             "subset of files based on their counts from the first of january of the same year.")
    parser.add_argument(
        '-ed','--ending_day_of_year', required=False, type=int, default=365,
        help="max threshold value for comparison with scenedate of files for considering a limited "
             "subset of files based on their counts from the first of january of the same year.")
    parser.add_argument(
        '-p','--precentage_of_training', required=False, type=between_zero_and_one_float_type,
        help="percentage of IDs that should be considered as training data (between 0,1). "
             "'1-precentage_of_training' fraction of data is considered as validation data.")
    parser.add_argument(
        '-bs','--batch_size', required=False, type=int,
        help="batch size for data generator")
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

    if parser.prog == "build_dataset.py" and (
        "shuffle_on_epoch_end" in arg
        or "shuffle_for_training" in arg
        or "memory_mode" in arg
        or "batch_size" in arg):
        parser.error("""For data building, none of "shuffle_on_epoch_end", "shuffle_for_training",
        "memory_mode" or "batch_size" should be appeared in th arguments.""")
    if not arg.apply_instead_of_training and arg.memory_mode:
        parser.error("Training mode should always be executed in file-base manner, not memory-based"
                     " one. Please remove both '-m' and '-i' from arguments for training purposes.")
    if not arg.apply_instead_of_training and int(arg.precentage_of_training) == 1:
        parser.error("Training mode should always be executed with 'precentage_of_training' less"
                     " than 1 .Please correct the value in arguments in order to consider some "
                     "validation data as well.")
    if arg.apply_instead_of_training and arg.shuffle_for_training:
        parser.error("Inference mode should always be executed without 'shuffle_for_training' arg.")
    if arg.apply_instead_of_training and arg.shuffle_on_epoch_end:
        parser.error("Inference mode should always be executed without 'shuffle_on_epoch_end' arg.")
    if arg.apply_instead_of_training and int(arg.precentage_of_training) != 1:
        parser.error("Inference mode should always be executed with 'precentage_of_training=1'. "
                     "Please correct the value in arguments.")
    if arg.apply_instead_of_training and (
                                    arg.beginning_day_of_year != 0 or arg.ending_day_of_year != 365
                                         ):
        parser.error("Inference mode should always be executed regardless of scenedate. Please remove"
            " 'beginning_day_of_year' and 'ending_day_of_year' from arguments in inference mode.")
    if arg.window_size % arg.aspect_ratio:
        parser.error(f"Window size must be dividable to value of aspect_ratio ={arg.aspect_ratio}")
    if arg.stride % arg.aspect_ratio:
        parser.error(f"Stridemust be dividable to value of aspect_ratio = {arg.aspect_ratio}")
    window_size_amsr2 = (arg.window_size // arg.aspect_ratio, arg.window_size // arg.aspect_ratio)
    stride_ams2_size = arg.stride // arg.aspect_ratio
    window_size = (arg.window_size, arg.window_size)
    stride_sar_size = arg.window_size
    rm_swath = arg.rm_swath
    distance_threshold = arg.distance_threshold
    datapath = arg.input_dir
    outpath = arg.output_dir
    nersc = arg.noise_method
    step_sar = arg.step_resolution_sar
    step_output = arg.step_resolution_output
    apply_instead_of_training = arg.apply_instead_of_training
    memory_mode = arg.memory_mode
    shuffle_on_epoch_end = arg.shuffle_on_epoch_end
    shuffle_for_training = arg.shuffle_for_training
    precentage_of_training = arg.precentage_of_training
    beginning_day_of_year = arg.beginning_day_of_year
    ending_day_of_year = arg.ending_day_of_year
    batch_size = arg.batch_size
    aspect_ratio = arg.aspect_ratio
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
    return Archive(
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
        apply_instead_of_training,
        memory_mode,
        shuffle_on_epoch_end,
        shuffle_for_training,
        precentage_of_training,
        beginning_day_of_year,
        ending_day_of_year,
        batch_size,
        aspect_ratio
                   )

class Configure():

    input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
    output_var_name = 'CT'
    amsr2_var_names = ['btemp_6_9h','btemp_6_9v']

    def __init__(self, archive):
        self.archive = archive
        self.batch_size = archive.batch_size
        self.BEGINNING_DAY_OF_YEAR =  archive.beginning_day_of_year
        self.ENDING_DAY_OF_YEAR =  archive.ending_day_of_year
        self.shuffle_on_epoch_end = archive.shuffle_on_epoch_end
        self.shuffle_for_training = archive.shuffle_for_training
        self.precentage_of_training = archive.precentage_of_training
        self.DATAPATH = archive.DATAPATH
        self.OUTPATH = archive.OUTPATH
        self.WINDOW_SIZE = archive.WINDOW_SIZE
        self.WINDOW_SIZE_AMSR2 = archive.WINDOW_SIZE_AMSR2
        self.ASPECT_RATIO = archive.ASPECT_RATIO

    def setup_generator(self):
        """
        This method is for setting up the generator based on the args of input and the config.
        Four steps of it are divided in four functions. ID list is the list of IDs that consists of
        All training and validation data for ML purposes.
        ID list will be the value corresponding to '_locs' of PROP dictionary for memory-based
        config. It will be the name of the 'npz' files files for file-based config.
        """
        self.filling_id_list()
        self.divide_id_list_into_partition()
        self.calculate_dims()
        self.instantiate_generator_with_params_and_associated_partition()


    # Design model
    def create_model(self):
        """
        create a keras model based on self.params variable (just for setting the input size)
        """
        input_ = layers.Input(shape=self.params['dims_input'])
        input_2 = layers.Input(shape=self.params['dims_amsr2'])
        x = layers.BatchNormalization()(input_)
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.AveragePooling2D(pool_size=(7,7), strides=(6,6))(x)
        x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = layers.Concatenate()([x, input_2])
        x = layers.UpSampling2D(size=(10, 10))(x)
        x = layers.UpSampling2D(size=(5, 5))(x)
        #x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same')(x)
        #x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same')(x)
        x = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(x)

        self.model = Model(
            inputs=[input_, input_2], outputs=x)
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
            )
        self.model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

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
                       callbacks=[tensorboard_callback, cp_callback],
                      )
        self.model.save("final_model")

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

    def divide_id_list_into_partition(self):
        """
        divide the id list into 'training' and 'validation' partition with the help of
        'precentage_of_training'.
        """
        if self.shuffle_for_training:
            random.shuffle(self.id_list)
        train_sublist_id_list = self.id_list[: int(len(self.id_list) * self.precentage_of_training)]
        valid_sublist_id_list = self.id_list[int(len(self.id_list) * self.precentage_of_training) :]
        print(f"total number of training samples: {len(train_sublist_id_list)}")
        print(f"total number of validation samples: {len(valid_sublist_id_list)}")
        # Datasets
        self.partition = {'train': train_sublist_id_list, 'validation': valid_sublist_id_list}

    def instantiate_generator_with_params_and_associated_partition(self):
        """
        create 'self.params' and instantiate the generator with proper partition and 'self.params'.
        """
        self.params = {'dims_input':      (*self.dims_input, len(self.input_var_names)),
                       'dims_output':     (*self.dims_output, 1),
                       'dims_amsr2':      (*self.dims_amsr2, len(self.amsr2_var_names)),
                       'output_var_name': self.output_var_name,
                       'input_var_names': self.input_var_names,
                       'amsr2_var_names': self.amsr2_var_names,
                       'batch_size':      self.batch_size,
                       'shuffle_on_epoch_end': self.shuffle_on_epoch_end,
                       'prop': self.archive.PROP,
                       }
        #generators
        self.training_generator = self.DataGenerator_(self.partition['train'], **self.params)
        self.validation_generator = self.DataGenerator_(self.partition['validation'], **self.params)

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

    def calc_dims(self):
        raise NotImplementedError('The calc_dims() method was not implemented')

    def filling_id_list(self):
        raise NotImplementedError('The filling_id_list() method was not implemented')


class FileBasedConfigure(Configure):
    DataGenerator_ = DataGeneratorFrom_npz_File
    extension = ".npz"

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


class MemoryBasedConfigure(Configure):
    DataGenerator_ = DataGeneratorFromMemory
    extension = ".nc"

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
        self.archive.calculate_PROP_of_archive(fil, basename(*self.list_of_names))
        self.id_list = self.archive.PROP['_locs']


    def instantiate_image_with_zeros_and_get_the_patch_locations_of_image(self):
        """
        Get the shape of amsr2 array by reading the '.nc' file. reconstructed image is in the size
        of amsr2 image multiplied by the aspect ratio.
        """
        shape_amsr2 = Dataset(*self.list_of_names)['btemp_6.9h'].shape
        self.img = np.zeros(shape=np.multiply(shape_amsr2, self.ASPECT_RATIO))
        self.patch_locs = self.archive.PROP['_locs']
