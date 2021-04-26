import numpy as np
from os import listdir
from os.path import isfile
from os.path import  join
import re
import datetime
import random
from keras.models import Sequential
from keras_classes import DataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

mypath="/workspaces/ASID-v2-builder/output"
beginning_day_of_year =  1#140
ending_day_of_year = 365
#import pandas as pd
only_npz = [f for f in listdir(mypath) if (f.endswith(".npz"))]
id_list = []
for x in only_npz:
    datetime_ = datetime.datetime.strptime(x.split("_")[0], '%Y%m%dT%H%M%S')
    first_of_jan_of_year = datetime.datetime(datetime_.year, 1, 1)
    if beginning_day_of_year <= (datetime_ - first_of_jan_of_year).days <= ending_day_of_year:
        id_list.append(x)
random.shuffle(id_list)
precentage_of_training = .8
train_sublist_id_list = id_list[: int(len(id_list) * precentage_of_training)]
valid_sublist_id_list = id_list[int(len(id_list) * precentage_of_training) :]
print(f"total number of training samples: {len(train_sublist_id_list)}")
print(f"total number of validation samples: {len(valid_sublist_id_list)}")
# Datasets
partition = {'train': train_sublist_id_list, 'validation': valid_sublist_id_list}

# Parameters
dims_input = np.load('/workspaces/ASID-v2-builder/output/' + id_list[0]).get('nersc_sar_primary').shape
dims_output = np.load('/workspaces/ASID-v2-builder/output/' + id_list[0]).get('CT').shape
params = {'dims_input': (*dims_input, 1),
          'dims_output': (*dims_output, 1),
          'batch_size': 4,
          'n_channels': 1,
          'shuffle': True}

#generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)
# Design model
input_ = layers.Input(shape=params['dims_input'])
# files that has built with normal noise
#input_ = layers.Input(shape=(*np.load('/workspaces/ASID-v2-builder/output/' + id_list[0] ).get('sar_primary').shape, 1))
x = layers.BatchNormalization()(input_)
x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same')(x)
x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same')(x)
x = layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)

model = Model(input_, x)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    )
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
model.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=40,
                    callbacks=[tensorboard_callback])
