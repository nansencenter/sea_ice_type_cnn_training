import argparse
import os
import random
from os.path import dirname, join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from archive import Archive

def parse_common_args():
    """Instantiates and creates common arguments of parser which works with 'argparse' of python."""
    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument()

    return parser

def between_zero_and_one_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f <= 0. or f > 1.:
        raise argparse.ArgumentTypeError("Argument must be =< " + str(1.) + " and > " + str(0.))
    return f

def type_for_nersc_noise(str_):
    """ Type function for argparse - a string that is specific to noise method """
    if not (str_=="" or str_=="nersc_"):
        raise argparse.ArgumentTypeError("'--noise_method' MUST be '' or 'nersc_'.")
    return str_

def common_parser():
    "Common parser which is shared between building the dataset and applying it"
    parser = argparse.ArgumentParser(description='Process the arguments of script')
    parser.add_argument('input_dir', type=str, help="Path to directory with input netCDF files")
    parser.add_argument(
        '-r', '--aspect_ratio', required=True, type=int,
        help="The ration between the cell size of primary and secondary input of ML model. stride"
        " and window_size must be dividable to it.")
    parser.add_argument(
        '-w', '--window_size', required=False, type=int,default=700,
        help="window size for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-swa','--rm_swath', required=False, type=int,default=0,
        help="threshold value for comparison with file.aoi_upperleft_sample to border the calculation")
    parser.add_argument(
        '-n', '--noise_method', required=False, type=type_for_nersc_noise, default="nersc_",
        help="the method that error calculation had been used for error.Leave as empty string '' for"
                    "ESA noise corrections or as 'nersc_' for the Nansen center noise correction.")
    parser.add_argument(
        '-d','--distance_threshold', required=False, type=int,default=0,
        help="threshold for distance from land in mask calculation")
    parser.add_argument(
        '-s', '--stride', required=False, type=int,default=700,
        help="stride for batching calculation(must be dividable to 50)")
    parser.add_argument(
        '-a','--step_resolution_sar', required=False, type=int,default=1,
        help="step for resizing the sar data")
    parser.add_argument(
        '-b','--step_resolution_output', required=False, type=int,default=1,
        help="step for resizing the output variables")
    return parser

def postprocess_the_args(arg):
    """
    postprocess the args based on the received values and return 'dict_for_archive_init'
    """
    if arg.window_size % arg.aspect_ratio:
        raise argparse.ArgumentError(arg.window_size,
                        f"Window size must be dividable to value of aspect_ratio ={arg.aspect_ratio}")
    if arg.stride % arg.aspect_ratio:
        raise argparse.ArgumentError(arg.stride,
                            f"Stride must be dividable to value of aspect_ratio = {arg.aspect_ratio}")
    window_size_amsr2 = (arg.window_size // arg.aspect_ratio, arg.window_size // arg.aspect_ratio)
    window_size = (arg.window_size, arg.window_size)
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
    nersc = arg.noise_method
    sar_names = [nersc + "sar_primary", nersc + "sar_secondary"]
    stride_ams2_size = arg.stride // arg.aspect_ratio
    dict_for_archive_init = {}
    dict_for_archive_init["sar_names"] = sar_names
    dict_for_archive_init["nersc"] = nersc
    dict_for_archive_init["datapath"] = arg.input_dir
    dict_for_archive_init["window_size"] = window_size
    dict_for_archive_init["window_size_amsr2"] = window_size_amsr2
    dict_for_archive_init["stride_sar_size"] = arg.stride
    dict_for_archive_init["stride_ams2_size"] = stride_ams2_size
    dict_for_archive_init["step_sar"] = arg.step_resolution_sar
    dict_for_archive_init["step_output"] = arg.step_resolution_output
    dict_for_archive_init["aspect_ratio"] = arg.aspect_ratio
    dict_for_archive_init["rm_swath"] = arg.rm_swath
    dict_for_archive_init["amsr_labels"] = amsr_labels
    dict_for_archive_init["distance_threshold"] = arg.distance_threshold
    return dict_for_archive_init

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

class Configure():
    create_model = None
    input_var_names = None
    output_var_name = None
    amsr2_var_names = None

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

    def calc_dims(self):
        raise NotImplementedError('The calc_dims() method was not implemented')

    def filling_id_list(self):
        raise NotImplementedError('The filling_id_list() method was not implemented')

    def predict_by_model(self):
        raise NotImplementedError('The predict_by_model() method was not implemented')

    def reconstruct_the_image_and_reset_archive_PROP(self):
        raise NotImplementedError('The reconstruct_the_image_and_reset_archive_PROP() method was not'
                                  ' implemented')

    def set_the_folder_of_reconstructed_files(self):
        raise NotImplementedError('The set_the_folder_of_reconstructed_files() method was not'
                                  ' implemented')

    def filling_id_list(self):
        raise NotImplementedError('The filling_id_list() method was not implemented')

    def apply_model(self):
        raise NotImplementedError('The apply_model() method was not implemented')

    def train_model(self):
        raise NotImplementedError('The train_model() method was not implemented')
