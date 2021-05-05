import argparse
import datetime
import os
import random
from os.path import basename

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from archive import Archive
from data_generator import DataGenerator


def read_input_params():
    """
    read the input data based on the command line arguments and return an instance of archive class
    """
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


def calculate_generator(only_npz,shuffle_on_epoch_end,beginning_day_of_year = 1,ending_day_of_year = 365,
                        precentage_of_training=.8, shuffle_for_training = True):
    """
    Find out the proper files based on "beginning_day_of_year" and "ending_day_of_year" among the
    "only_npz" files. Then make the training and validation list of samples from the only_npz list.

    Running the model by linking to this generator will provide the data of training and validation
    batch by batch with batch size of this generator for the model.

    "shuffle_for_training", "shuffle_on_epoch_end", and "precentage_of_training" are configures for
    setting two sublists of training and validation out of only_npz list.
    """
    batch_size = 4
    id_list = []
    for x in only_npz:
        datetime_ = datetime.datetime.strptime(basename(x).split("_")[0], '%Y%m%dT%H%M%S')
        first_of_jan_of_year = datetime.datetime(datetime_.year, 1, 1)
        if beginning_day_of_year <= (datetime_ - first_of_jan_of_year).days <= ending_day_of_year:
            id_list.append(x)
    if shuffle_for_training:
        random.shuffle(id_list)
    train_sublist_id_list = id_list[: int(len(id_list) * precentage_of_training)]
    valid_sublist_id_list = id_list[int(len(id_list) * precentage_of_training) :]
    print(f"total number of training samples: {len(train_sublist_id_list)}")
    print(f"total number of validation samples: {len(valid_sublist_id_list)}")
    # Datasets
    partition = {'train': train_sublist_id_list, 'validation': valid_sublist_id_list}
    input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
    output_var_name = 'CT'
    amsr2_var_names = ['btemp_6_9h','btemp_6_9v']
    # obtaining the shape from the first sample of data
    dims_input = np.load(id_list[0]).get(input_var_names[0]).shape
    dims_output = np.load( id_list[0]).get(output_var_name).shape
    dims_amsr2 = np.load(id_list[0]).get(amsr2_var_names[0]).shape
    params = {'dims_input': (*dims_input, len(input_var_names)),
            'dims_output': (*dims_output, 1),
            'dims_amsr2': (*dims_amsr2, len(amsr2_var_names)),
                'output_var_name':output_var_name,
                'input_var_names':input_var_names,
                'amsr2_var_names':amsr2_var_names,
            'batch_size': batch_size,
            }
    #generators
    training_generator = DataGenerator(partition['train'],shuffle_on_epoch_end, **params)
    validation_generator = DataGenerator(partition['validation'],shuffle_on_epoch_end, **params)
    return training_generator, validation_generator, params

# Design model
def create_model(params):
    """
    create a keras model based on params variable (just for setting the input size) and return it
    """
    input_ = layers.Input(shape=params['dims_input'])
    input_2 = layers.Input(shape=params['dims_amsr2'])
    # files that has built with normal noise
    #input_ = layers.Input(shape=(*np.load('/workspaces/ASIP-v2-builder/output/' + id_list[0] ).get('sar_primary').shape, 1))
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

    model = Model(
        inputs=[input_,input_2], outputs=x)
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model
