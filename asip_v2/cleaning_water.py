import glob
import os
import random as rd
import json
import shutil
import numpy as np


idir = '/Data/preprocessing4hugo/output'
idir_w_out = '/Data/preprocessing4hugo/output/output_out_w'
water_threshold =0.2

os.makedirs(idir_w_out, exist_ok=True)


with open(f'{idir}/processed_files.json') as fichier_json:
    all_nc = json.load(fichier_json)
sum_=0
out=0
keep=0
for nc in all_nc :
    name = nc[:15]
    files = sorted(glob.glob(f'{idir}/{name}/*.npz'))
    sum_+=len(files)
    for npz in files :
        vector_param = np.load(npz).get("ice_type")
        if vector_param[0] == 2 : #it is water
            prob = rd.random()
            # we keep only 20 % of water batches
            if prob >= water_threshold :
                out += 1
                os.makedirs(f'{idir_w_out}/{name}', exist_ok=True)
                shutil.move(npz, f'{idir_w_out}/{name}/{npz[-10:]}')
            else:
                keep += 1
print(sum_)
print(out)
print(keep)
print(out+keep)