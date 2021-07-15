import argparse
import os
import random
from os.path import dirname, join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from archive import Archive

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
    parser.add_argument('input-dir', type=str, help="Directory with input netCDF files")
    parser.add_argument('output-dir', type=str, help="Directory for output files (npz files)",)
    parser.add_argument(
        '-w', '--window', required=False, type=int, default=256,
        help="window size for selecting patches from SAR/ice charts")
    parser.add_argument(
        '-w2', '--window2', required=False, type=int, default=16,
        help="window size for selecting patches from AMSR2")
    parser.add_argument(
        '-swa','--rm_swath', required=False, type=int, default=0,
        help="threshold value for comparison with file.aoi_upperleft_sample to border the calculation")
    parser.add_argument(
        '-n', '--name_sar', required=False, default="nersc_sar", choices=["nersc_sar", "sar"],
        help="Prefix for SAR band name")
    parser.add_argument(
        '-d','--distance-threshold', required=False, type=int, default=0,
        help="threshold for distance from land in mask calculation")
    parser.add_argument(
        '-s', '--stride', required=False, type=int, default=None,
        help="stride for selecting patches from SAR/ice charts")
    parser.add_argument(
        '-r','--resize-step', required=False, type=int, default=1,
        help="step for resizing the SAR/ice charts data")
    return parser

def postprocess_the_args(arg):
    """
    postprocess the args based on the received values and return 'dict_for_archive_init'
    """
    amsr2_names = [
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
    names_sar = [arg.name_sar + "_primary", arg.name_sar + "sar_secondary"]
    if arg.stride:
        stride = arg.stride
    else:
        stride = arg.window

    stride_amsr2 = arg.window2 * arg.stride / arg.window

    dict_for_archive_init = dict(
        input_dir = arg.input_dir,
        output_dir=arg.output_dir,
        names_sar = sar_names,
        names_amsr2 = amsr2_names,
        window_sar = arg.window,
        window_amsr2 = arg.window2,
        stride_sar = stride,
        stride_amsr2 = stride,
        resample_step_amsr2 = arg.window / arg.window2,
        resize_step_sar = arg.resize_step,
        rm_swath = arg.rm_swath,
        distance_threshold = arg.distance_threshold,
    )
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
        self.percentage_of_training = archive.percentage_of_training
        self.DATAPATH = archive.DATAPATH
        self.OUTPATH = archive.OUTPATH
        self.WINDOW_SIZE = archive.WINDOW_SIZE
        self.WINDOW_SIZE_AMSR2 = archive.WINDOW_SIZE_AMSR2
        self.ASPECT_RATIO = archive.ASPECT_RATIO

    def setup_generator(self):
        """
        This method is for setting up the generator based on the args of input and the config.
        Five steps of it are divided in five functions. ID list is the list of IDs that consists of
        All training and validation data for ML purposes.
        ID list will be the value corresponding to '_locs' of PROP dictionary for memory-based
        config. It will be the name of the 'npz' files files for file-based config.
        """
        self.filling_id_list()
        self.divide_id_list_into_partition()
        self.calculate_dims()
        self.set_params()
        self.instantiate_generators_with_associated_partition()

    def divide_id_list_into_partition(self):
        """
        divide the id list into 'training' and 'validation' partition with the help of
        'percentage_of_training'.
        """
        if self.shuffle_for_training:
            random.shuffle(self.id_list)
        train_sublist_id_list = self.id_list[: int(len(self.id_list) * self.percentage_of_training)]
        valid_sublist_id_list = self.id_list[int(len(self.id_list) * self.percentage_of_training) :]
        print(f"total number of training samples: {len(train_sublist_id_list)}")
        print(f"total number of validation samples: {len(valid_sublist_id_list)}")
        # Datasets
        self.partition = {'train': train_sublist_id_list, 'validation': valid_sublist_id_list}

    def set_params(self):
        """
        create 'self.params'.
        """
        self.params = {'dims_input':      (*self.dims_input, len(self.input_var_names)),
                       'dims_output':     (*self.dims_output, 1),
                       'dims_amsr2':      (*self.dims_amsr2, len(self.amsr2_var_names)),
                       'output_var_name': self.output_var_name,
                       'input_var_names': self.input_var_names,
                       'amsr2_var_names': self.amsr2_var_names,
                       'batch_size':      self.batch_size,
                       'shuffle_on_epoch_end': self.shuffle_on_epoch_end,
                       }

    def instantiate_generators_with_associated_partition(self):
        #generators
        self.training_generator = self.DataGenerator_(self.partition['train'], **self.params)
        self.validation_generator = self.DataGenerator_(self.partition['validation'], **self.params)

    def calculate_dims(self):
        raise NotImplementedError('The calculate_dims() method was not implemented')

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
