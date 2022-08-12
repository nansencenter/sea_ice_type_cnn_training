# This script is used to train and calculate the metrics for the sod network 

input_dir_json = '/tf/data/hugo_sod/'
idir = '/tf/data/hugo_sod/output_preprocessed/'

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
import seaborn as sebrn
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

        
with open(f'{input_dir_json}/vector_combinations.json') as fichier_json:
    list_combi = json.load(fichier_json)['all_work_comb']
    
def create_model():
    """
    create a keras model 
    """
        # number of ice classes
    nbr_classes = len(list_combi)  
    # size of SAR subimages
    ws = 50
    # size of AMRS2 subimages
    ws2 = 10
    # size of convolutional filters
    cs = 3
    # number of filters per convolutional layer (x id)
    c1,c2,c3 = 32,64,64
    # number of neurons per hidden neural layer number (x id)
    n1,n2,n3 = 64,32,16
    # value of dropout
    dropout_rate = 0.1
    # value of L2 regularisation
    l2_rate = 0.001
    
    
    input_ = layers.Input(shape =(ws, ws, 4) )
    input_2 = layers.Input(shape = (ws2, ws2, 14))
    
    x = layers.BatchNormalization()(input_)
    x = layers.Conv2D(c1, (cs, cs),  activation='relu')(x)
    x = layers.Conv2D(c1, (cs, cs), padding ='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = layers.Conv2D(c2, (cs, cs), activation='relu')(x)
    x = layers.Conv2D(c2, (cs, cs), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = layers.BatchNormalization()(x)
    input_2 = BatchNormalization()(input_2)


    x = layers.Concatenate()([x, input_2])
    
    x = layers.Conv2D(c3, (cs, cs), activation='relu')(x)
    x = layers.Conv2D(c3, (cs, cs), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    
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

training_generator = DataGenerator_sod_f(train_files, **params)
validation_generator = DataGenerator_sod_f(valid_files, **params)


# creation of the model 
model = create_model()
#callbacks
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model_sod_50bis_nopatience', 
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
                    epochs=50, 
                    callbacks=[mc])
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig("history_model_sod_50bis_nopatience")
plt.show()

#metrics
y_pred = model.predict(validation_generator)
y_val = np.vstack([vg[1] for vg in validation_generator])
rmse_matrix=np.empty((32,32))
pearson_matrix = np.empty((32,32))
for id_class_pred in range(y_pred.shape[1]):
    classes_pred = y_pred[:,id_class_pred]
    for id_class_val in range (y_val.shape[1]):
        classes_val = y_val[:,id_class_val]
        rmse = mean_squared_error(classes_val, classes_pred)
        rmse_matrix[id_class_pred][id_class_val] = rmse
        pearson_value = stats.pearsonr(classes_val, classes_pred)
        pearson_matrix[id_class_pred][id_class_val] = pearson_value[0]

        
        
        
print('rmse_matrix')
rm=[]
for i in rmse_matrix :
    l=[]
    for j in i :
        l.append(np.round(j,8))
    rm.append(l)
print(rm)

    
print('pearson_matrix')
pm=[]
for i in pearson_matrix :
    l=[]
    for j in i :
        l.append(np.round(j,8))
    pm.append(l)
print(pm)



y_ticks= [(i+0.5) for i in range (len(list_combi))]


plt.clf()
plt.figure(figsize=(18,18))
fx = sebrn.heatmap(pearson_matrix, annot=True, cmap='bwr', fmt=".3f", cbar=False)
fx.set_title('Pearson correlation Matrix \n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('True Values ')
fx.xaxis.set_ticklabels(list_combi)
fx.xaxis.tick_top()
fx.set_yticks(y_ticks)
fx.yaxis.set_ticklabels(ticklabels=list_combi)
plt.savefig('Pearson_matrix_sod_50bis_nopatience')
plt.show()




# Using Seaborn heatmap to create the plot
plt.clf()
plt.figure(figsize=(18,18))
fx = sebrn.heatmap(rmse_matrix, annot=True, cmap='Blues', fmt=".3f", cbar=False)
fx.set_title('RMSE Matrix \n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('True Values ')
fx.xaxis.set_ticklabels(list_combi)
fx.xaxis.tick_top()
fx.set_yticks(y_ticks)
fx.yaxis.set_ticklabels(ticklabels=list_combi)
plt.savefig('rmse_matrix_sod_50bis_nopatience')
plt.show()



interval = time.time() - start_time
print ('Total time in seconds:'+ str(interval))