
from netCDF4 import Dataset
from os.path import join, isdir, dirname
from os import listdir, mkdir
import os
from data_builder import read_input_params
from keras_script import calculate_generator, create_model
import tensorflow as tf
import numpy as np

def main():

    stride = 700
    ws = 700 #window size
    raise ValueError("stride must be equal or greater than the window size") if stride<ws else None
    outputpath="/workspaces/ASID-v2-builder/output"
    netcdfpath="/workspaces/ASID-v2-builder"
    reconstruct_path = join(dirname(outputpath), "reconstructs_folder")
    if not isdir(reconstruct_path):
        mkdir(reconstruct_path)

    scene_dates = {f.split("_")[0] for f in listdir(outputpath) if f.endswith(".npz")}
    for scene_date in scene_dates:
        only_npzs_of_scene_date = [
                    join(outputpath, f) for f in listdir(outputpath) if (f.startswith(scene_date))
                                 ]
        abs_add_netcdf = join(
                        netcdfpath, *{x for x in listdir(netcdfpath) if x.startswith(scene_date)}
                             )
        inference_generator, _ , params = calculate_generator(only_npz = only_npzs_of_scene_date,
                                                              shuffle_on_epoch_end = False,
                                                              beginning_day_of_year =  1,
                                                              ending_day_of_year = 365,
                                                              precentage_of_training = 1,
                                                              shuffle_for_training = False
                                                             )

        model = create_model(params)
        latest = tf.train.latest_checkpoint("models")
        model.load_weights(latest)

        y_pred = model.predict(inference_generator, verbose=0, steps=None, callbacks=None,
                                max_queue_size=10, workers=1, use_multiprocessing=False)
        shape_amsr2 = Dataset(abs_add_netcdf)['btemp_6.9h'].shape
        img = np.zeros(shape=np.multiply(shape_amsr2, 50))
        #each locations based on each file name
        locs = [(x.split("-")[-1].split(".")[0]).split("_") for x in only_npzs_of_scene_date]
        # convert them into integer
        locs = [(int(x[0]), int(x[1])) for x in locs]
        locs = np.multiply(locs, stride)
        for i in range(y_pred.shape[0]):
            img[locs[i][0]:locs[i][0] + ws, locs[i][1]:locs[i][1] + ws] = y_pred[i,:,:,0]
        np.savez(join(reconstruct_path, f"{scene_date}_reconstruct.npz"), img)

if __name__ == "__main__":
    main()
