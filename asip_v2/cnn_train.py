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
from data_generator import HugoDataGenerator, DataGenerator_sod_f, HugoBinaryGenerator

import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn as sk
from sklearn.metrics import confusion_matrix

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

idir = '/Data/preprocessing4hugo/output/'
input_dir_json = '/Data/'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
        
def create_model():
    """ Create sequential CNN with convolutional and dense layers
    
    """
    # number of ice classes
    nbr_classes = 4  
    # size of SAR subimages
    ws = 50
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

    model = Sequential()

    # Convolutional part
    model.add(BatchNormalization(input_shape=(ws, ws, 2)))
    model.add(Conv2D(c1, (cs, cs), activation='relu'))
    model.add(MaxPooling2D((2, 2),2))
    model.add(Conv2D(c2, (cs, cs), activation='relu'))
    model.add(Conv2D(c3, (cs, cs), activation='relu'))
    model.add(MaxPooling2D((2, 2),2))
    model.add(BatchNormalization())

    model.add(Flatten())

    # Neural network part (hidden layers)
    model.add(Dropout(dropout_rate))
    model.add(Dense(n1, kernel_regularizer=l2(l2_rate), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n2, kernel_regularizer=l2(l2_rate), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n3, kernel_regularizer=l2(l2_rate), activation='relu'))
    model.add(Dropout(dropout_rate))

    # Last neural layer (not hidden)
    model.add(Dense(nbr_classes, kernel_regularizer=l2(l2_rate), activation='softmax'))
    
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
amsr2_var_names = [ 'btemp_6_9h',
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

training_generator = HugoBinaryGenerator(train_files, **params)
validation_generator = HugoBinaryGenerator(valid_files, **params)

# training_generator = HugoDataGenerator(train_files, **params)
# validation_generator = HugoDataGenerator(valid_files, **params)

# training_generator = DataGenerator_sod_f(train_files, **params)
# validation_generator = DataGenerator_sod_f(valid_files, **params)


# creation of the model 
model = create_model()
#callbacks
mc = tf.keras.callbacks.ModelCheckpoint(filepath='hugo_model_test_all_remove', 
                                        monitor='val_loss',
                                        verbose=1, 
                                        save_best_only=True,
                                        mode='min')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#optimizers and compile
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy')

#fit
model.summary()
history = model.fit(training_generator, 
                    use_multiprocessing=True,
                    workers=8,
                    validation_data=validation_generator,
                    epochs=20, 
                    callbacks=[mc, es])
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig("history_remove")
plt.show()

#confusion matrix
y_pred = model.predict(validation_generator)
y_val = np.vstack([vg[1] for vg in validation_generator])
y_val_index =[]
y_pred_index =[]
for val, pred in zip(y_val, y_pred):
    if max(pred)>0.5 :
        y_val_index.append(np.argmax(val))
        y_pred_index.append(np.argmax(pred))
cm = confusion_matrix(y_val_index, y_pred_index)
print(cm)
fig1 = plt.figure()
plt.imshow(cm)
plt.savefig('confusion_matrix_remove')
plt.show()
interval = time.time() - start_time
print ('Total time in seconds:'+ str(interval))