import numpy as np
from os import listdir
from os.path import isfile, join
import re
import datetime

from keras.models import Sequential
from keras_classes import DataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from utility import calculate_generator, create_model

def main():
    outputpath="/workspaces/ASIP-v2-builder/output"

    only_npz = [join(outputpath, f) for f in listdir(outputpath) if (f.endswith(".npz"))]

    training_generator, validation_generator, params = calculate_generator(
    only_npz = only_npz,
    shuffle_on_epoch_end = True,
    beginning_day_of_year =  1,
    ending_day_of_year = 365,
    precentage_of_training = .8,
    shuffle_for_training = True
    )

    model = create_model(params)
    model.summary()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = join("models",datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+"{epoch:04d}"),
        verbose = 1,
        save_weights_only=True,
        save_freq=6*params["batch_size"])
    # Train model on dataset
    model.fit(training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=4,
                        callbacks=[tensorboard_callback, cp_callback])
    model.save("final_model")

if __name__ == "__main__":
    main()
