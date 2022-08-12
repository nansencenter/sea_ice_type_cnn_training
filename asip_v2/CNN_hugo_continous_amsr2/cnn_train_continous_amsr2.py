#this script allows to perform the training and to calculate the metrics for the networks with AMSR2 data

input_dir_json = '/tf/data/hugo_continous_amsr2/output_preprocessed_continous/'
idir = '/tf/data/hugo_continous_amsr2/output_preprocessed_continous/output/'


import os
from os.path import basename, dirname, join
import time

start_time = time.time()

import glob
import argparse
import datetime
import random
import json
from netCDF4 import Dataset
import numpy as np

import data_generator
from data_generator import HugoDataGenerator, DataGenerator_sod_f, HugoBinaryGenerator, HugoSarDataGenerator, HugoAMRS2DataGenerator

import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn as sk

from sklearn.metrics import (confusion_matrix, 
                            mean_squared_error, 
                            accuracy_score,
                            precision_score,
                            recall_score)
from scipy import stats
import statistics
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,
                                     Flatten,
                                     Dropout,
                                     BatchNormalization, 
                                     Conv2D, 
                                     MaxPooling2D)
from tensorflow.keras.regularizers import l2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
        
def create_model():
    """
    create a keras model 
    """
        # number of ice classes
    nbr_classes = 4  
    # size of SAR subimages
    ws = 50
    # size of AMRS2 subimages
    ws2 = 10
    # size of convolutional filters
    cs = 3
    # number of filters per convolutional layer (x id)
    c1,c2,c3 = 32,32,32
    # number of neurons per hidden neural layer number (x id)
    n1,n2,n3 = 16,16,64
    # value of dropout
    dropout_rate = 0.1
    # value of L2 regularisation
    l2_rate = 0.001
    
    
    input_ = layers.Input(shape =(ws, ws, 4) )
    input_2 = layers.Input(shape = (ws2, ws2, 14))
    
    x = layers.BatchNormalization()(input_)
    x = layers.Conv2D(c1, (cs, cs),  activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = layers.Conv2D(c2, (cs, cs), activation='relu')(x)
    x = layers.Conv2D(c3, (cs, cs), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = layers.BatchNormalization()(x)
    input_2 = BatchNormalization()(input_2)


    x = layers.Concatenate()([x, input_2])
    
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n1, kernel_regularizer=l2(l2_rate), activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n2, kernel_regularizer=l2(l2_rate), activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n3, kernel_regularizer=l2(l2_rate), activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    # Last neural layer (not hidden)
    x = layers.Dense(nbr_classes, kernel_regularizer=l2(l2_rate), activation='softmax')(x)

    model = Model(inputs=[input_, input_2], outputs=x)
    opt = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

train_ratio = 0.7
with open(f'{idir}processed_files.json') as fichier_json:
    all_nc = json.load(fichier_json)
npz_files=[]

for nc in all_nc :
    name = nc[:15]
    files = sorted(glob.glob(f'{idir}/{name}/*.npz'))
    npz_files += files
random.shuffle(npz_files)


print('Files number : '+ str (len(npz_files)))
train_files_number = int(len(npz_files) * train_ratio)
train_files = npz_files[:train_files_number]
valid_files = npz_files[train_files_number:]

input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
amsr2_var_names = ['btemp_6_9h',
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
                   'btemp_89_0v'
                  ]

output_var_name = 'ice_type'
dims_amsr2 = np.load(npz_files[0])[amsr2_var_names[0]].shape


params = {'dims_amsr2':      (*dims_amsr2, len(amsr2_var_names)),
          'idir_json':       input_dir_json,
          'output_var_name': output_var_name,
          'input_var_names': input_var_names,
          'amsr2_var_names': amsr2_var_names,
          'batch_size':      50,
          'shuffle_on_epoch_end': False,
           }

training_generator = HugoAMRS2DataGenerator(train_files, **params)
validation_generator = HugoAMRS2DataGenerator(valid_files, **params)


# creation of the model 
model = create_model()
#callbacks
mc = tf.keras.callbacks.ModelCheckpoint(filepath='hugo_model_continous_amsr2_bis_all100_nopatience', 
                                        monitor='val_loss',
                                        verbose=1, 
                                        save_best_only=True,
                                        mode='min')

# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

#fit
model.summary()
history = model.fit(training_generator, 
                    use_multiprocessing=True,
                    workers=8,
                    validation_data=validation_generator,
                    epochs=100, 
                    callbacks=[mc])
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig("history_hugo_continous_amsrs2_bis_all100_nopatience")
plt.show()

#metrics
y_pred = model.predict(validation_generator)
y_val = np.vstack([vg[1] for vg in validation_generator])
rmse_matrix=np.empty((4,4))
pearson_matrix = np.empty((4,4))
for id_class_pred in range(y_pred.shape[1]):
    classes_pred = y_pred[:,id_class_pred]
    for id_class_val in range (y_val.shape[1]):
        classes_val = y_val[:,id_class_val]
        rmse = mean_squared_error(classes_val, classes_pred)
        rmse_matrix[id_class_pred][id_class_val] = rmse
        pearson_value = stats.pearsonr(classes_val, classes_pred)
        pearson_matrix[id_class_pred][id_class_val] = pearson_value[0]

print('rmse_matrix')
print(rmse_matrix)
print('pearson_matrix')
print(pearson_matrix)

fig2 = plt.figure()
plt.imshow(rmse_matrix)
plt.savefig('rmse_matrix_hugo_continous_amsr2_bis_all100_nopatience')
plt.show()

fig3 = plt.figure()
plt.imshow(pearson_matrix)
plt.savefig('pearson_matrix_hugo_continous_amsr2_bis_all100_nopatience')
plt.show()


interval = time.time() - start_time
print ('Total time in seconds:'+ str(interval))