import glob
import argparse
import datetime
import os
from os.path import basename, dirname, join
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


from data_generator import DataGeneratorFrom_npz_File

idir = '/data1/antonk/tmp'

train_ratio = 0.7
npz_files = sorted(glob.glob(f'{idir}/*npz'))
random.shuffle(npz_files)
train_files_number = int(len(npz_files) * train_ratio)
train_files = npz_files[:train_files_number]
valid_files = npz_files[train_files_number:]

input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
amsr2_var_names = [
    'btemp_6_9h',
    'btemp_6_9v',
    'btemp_7_3h',
    'btemp_7_3v',
    'btemp_10_7h',
    'btemp_10_7v',
    'btemp_18_7h',
    'btemp_18_7v',
    'btemp_23_8h',
    'btemp_23_8v',
    'btemp_36_5h',
    'btemp_36_5v',
    'btemp_89_0h',
    'btemp_89_0v',
 ]
output_var_name = 'ice_type'

dims_input = np.load(npz_files[0])[input_var_names[0]].shape
dims_output = np.load(npz_files[0])[output_var_name].shape
dims_amsr2 = np.load(npz_files[0])[amsr2_var_names[0]].shape

params = {'dims_input':      (*dims_input, len(input_var_names)),
           'dims_output':     (*dims_output,),
           'dims_amsr2':      (*dims_amsr2, len(amsr2_var_names)),
           'output_var_name': output_var_name,
           'input_var_names': input_var_names,
           'amsr2_var_names': amsr2_var_names,
           'batch_size':      50,
           'shuffle_on_epoch_end': False,
           }

training_generator = DataGeneratorFrom_npz_File(train_files, **params)

print(
    training_generator[0][0][0].shape,
    training_generator[0][0][1].shape,
    training_generator[0][1].shape,
)


input_ = layers.Input(shape=params['dims_input'])
input_2 = layers.Input(shape=params['dims_amsr2'])
x = layers.BatchNormalization()(input_)
x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Concatenate()([x, input_2])
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(x)
x = layers.Conv2D(filters=4, kernel_size=1, strides=1, padding='same', activation='softmax')(x)

model = Model(inputs=[input_, input_2], outputs=x)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    )
model.compile(optimizer=opt, loss='categorical_crossentropy')
model.summary()

model.fit(training_generator, epochs=2)
