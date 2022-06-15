import glob
import json
import numpy as np

idir='/Data/analysis'
with open(f'{idir}/output/processed_files.json') as fichier_json:
    all_nc = json.load(fichier_json)

for nc in all_nc :
    name = nc[:15]
    npz_files = sorted(glob.glob(f'{idir}/output/{name}*.npz'))
    array = []
    for npz_file in npz_files:
        d = np.load(npz_file)
        array.append(np.hstack([d["ice_type"], d["distance_border"]]))
    array = np.array(array)
    np.savez(f'{idir}/output_summary/{name}_all_npz.npz', array=array)