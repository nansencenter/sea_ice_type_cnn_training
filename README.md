# description
This repository provides the facilities for ASIP v2 data [(webpage)](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134). Manual of this dataset is provided [here](https://data.dtu.dk/ndownloader/files/24951176).
### DISCLAIMER: This code project is released as it without guarantees and extensive testing. It is meant to guide and help researchers and students get started on sea ice modelling with Convolutional neural networks.
The order of execution of different parts of the code is as follow:
 1. [Execute the data building](#requirement)
 2. [Execute the tensorflow training](#execute-the-data-building)
 3. [Execute the inference (apply) code](#execute-the-inference-code)
 4. [Plotting the result of inference](#plotting-the-result-of-inference)

# Requirement
just run the following command in your environment in order to install the requirements:
```python
pip install -r requirements.txt
```
The users of Microsoft VScode can easily open the remote development container with the help of `.devcontainer` folder and Dockerfile
# Usage
This code uses [python argparse](https://docs.python.org/3/library/argparse.html) for reading the input from command line.
Each of three scripts including `train_model.py`, `apply_model.py` and `build_dataset.py` can show which argument belongs to them by placing `-h` after their name. For example, **python train_model.py -h** shows the ones that belong to training activities.
# Execute the data building
By just giving the full absolute address of the folder that contains all of the uncompressed .nc files of ASIP data, data building part of code is able to build the data based on those files and make them ready for further Machine learning training.

This can be done by writing this command for training purposes:

```python
python build_dataset.py /absolute/path/to/the/folder/of/input_files
```


Only the unmasked locations of data are selected among the others for having a completely clean data in order to feed it for ML training.

>  HINT: The folder containing input files must have all of the **.nc** files without any subfoldering

This command will create a folder named **output** in the folder that contains the `build_dataset.py` and write all the output files into it.


As an example, for the case of building data from **/fold1** folder and store them in **/fold2** folder with nersc noise calculation, and having both window size and stride of **400**, the command below is used:

```python
python build_dataset.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400
```
Table below shows how the arguments are working:
___
| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  []                 |  []                 |  [no default]| The first and the only positional argument is the path to directory with input netCDF files needed for data building or applying the trained model (inference)
|  -o                 |  --output_dir       | [no default]      |Path to directory for output of building (.npz files)
|  -n                 |    --noise_method   |'nersc_'|the method that error calculation had  been used for error. Leave as empty string '' for ESA noise corrections or as 'nersc_' for the Nansen center noise correction.
|  -w                 |   --window_size     |  700 | window size (of sar and ice chart data) for batching calculation (must be dividable to aspect ratio,the ratio between the cell size of primary and secondary input of network)(This will be the size of image samples that has been used for ML training step)
|  -s                 |   --stride          |  700 | stride (of sar and ice chart data) for batching calculation (must be dividable to aspect ratio,the ratio between the cell size of primary and secondary input of network)(This will be the stride that determines the overlapping areas between image samples for ML training step)
|  -r                 |   --aspect_ratio     |  50 | The ration between the cell size of primary and secondary input of ML model. stride and window_size must be dividable to it.
|  -swa                 |   --rm_swath        |  0    |threshold value for comparison with netCDF file.aoi_upperleft_sample to border the calculation
|  -d                 |   --distance_threshold |  0  |threshold for distance from land in mask calculation
|  -a                 |   --step_resolution_sar | 1  |step for resizing the sar data (default value leads to no resizing)
|  -b                 |   --step_resolution_output|1 |step for resizing the ice chart data (default value leads to no resizing)
___


# Execute the tensorflow training
After building the data, you can train the tensorflow model with those `.npz` files as the result of
data building calculation. To do this, run the script `train_model.py` by setting the address of output folder (which has been used with training mode) from pervious calculation (data building) with '-o' in the arguments.

> It is strongly recommend to read the link below before using this part of the code because everything for file based config (including the classes and scripts) is developed based on explanation of this web page:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


If you want to run the training with scenes that are belong to a specific season of the year(spring,summer,etc), then you can set `beginning_day_of_year` and `ending_day_of_year` variable in the arguments of command line in order to make use of the files that are only belong to the period of year between these two numbers. These two numbers are start and end day count from the beginning of the year for reading data between them.


Train the tensorflow can be done by writing this command in order to training from npz files of *fold2* folder:

```python
python train_model.py -o /fold2 -bs 4 -p 0.8 -see -sft
```
In the above example the npz files are being read from `/fold2` folder.

Table below shows how the arguments are working:
___
| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  -o                 |  --output_dir       | [no default]   |Path to directory with output files (.npz files as the output of building data)
|  -see                |   --shuffle_on_epoch_end  |  False (in the case of absence in the arguments) | Flag for Shuffling the training subset of IDs at the end of every epoch during the training
|  -sft                |   --shuffle_for_training  |  False (in the case of absence in the arguments) | Flag for Shuffling the list of IDs before dividing it into two 'training' and 'validation' subsets
|  -bd                |   --beginning_day_of_year          |  0 | min threshold value for comparison with scenedate of files for considering a limited subset of files based on their counts from the first of january of the same year
|  -ed                |   --ending_day_of_year          |  365 | max threshold value for comparison with scenedate of files for considering a limited subset of files based on their counts from the first of january of the same year
|  -p                |   --percentage_of_training          |  [] | percentage of IDs that should be considered as training data (between 0,1). '1-percentage_of_training' fraction of data is considered as validation data.
|  -bs               |   --batch_size          |  [] | batch size for data generator
___

# Execute the inference code
The output of the ML network is the patches of image, not the whole image with original size. For seeing the result of network (as a whole image,i.e. a scene) after training, `apply_model.py` can be used. To do this, just like previous example of command line of training, we can run the below command in the command line:
```python
python apply_model.py /fold1 -n nersc_ -w 400 -s 400 -bs 4
```
> **Hint**: If you:
> * use resizing for building the data
> * use values of stride and window size in a way that there is a overlapping area for building and training.
>
> and then train the network, this `apply_model.py` code (and consequent plotting) is **`not`** applicable.
**This inference code is only for cases that resizing and overlapping is not used.**

This mode is executed in memory based manner. In this case, only the 'nc' files of /fold1 is taking into consideration for applying the trained model on them. The trained model is selected automatically by tensoeflow as the last trained model (configurable by `checkpoint` file; more information about the saving mechanism and checkpoint file is [here](https://www.tensorflow.org/guide/checkpoint)).

> Hint: it is important to give values of `window_size`, `stride` and `batch size` identical to those of data building calculation. Otherwise, applying the model is meaningless.

A folder named `reconstructs_folder` will be created at the same level of *output_dir* and the reconstructed files will be saved inside that folder.
___
Table below shows how the arguments are working:
| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  []                 |  []                 |  [no default]| The first and the only positional argument is the path to directory with input netCDF files needed for data building or applying the trained model (inference)
|  -n                 |    --noise_method   |'nersc_'|the method that error calculation had  been used for error. Leave as empty string '' for ESA noise corrections or as 'nersc_' for the Nansen center noise correction.
|  -w                 |   --window_size     |  700 | window size (of sar and ice chart data) for batching calculation (must be dividable to aspect ratio,the ratio between the cell size of primary and secondary input of network)(This will be the size of image samples that has been used for ML training step)
|  -s                 |   --stride          |  700 | stride (of sar and ice chart data) for batching calculation (must be dividable to aspect ratio,the ratio between the cell size of primary and secondary input of network)(This will be the stride that determines the overlapping areas between image samples for ML training step)
|  -r                 |   --aspect_ratio     |  50 | The ration between the cell size of primary and secondary input of ML model. stride and window_size must be dividable to it.
|  -swa                 |   --rm_swath        |  0    |threshold value for comparison with netCDF file.aoi_upperleft_sample to border the calculation
|  -d                 |   --distance_threshold |  0  |threshold for distance from land in mask calculation
|  -a                 |   --step_resolution_sar | 1  |step for resizing the sar data (default value leads to no resizing)
|  -b                 |   --step_resolution_output|1 |step for resizing the ice chart data (default value leads to no resizing)
|  -bs               |   --batch_size          |  [no default] | batch size for data generator
___
# Plotting the result of inference

For plotting, you can run a separate python script called `show.py`. You have to make sure that the dependencies are ready for this script. It means you have to install `scipy` and `numpy` on your env and run the `show.py`. This `show.py` code can also be substituted with an interactive jupyter-notebook.
Plotting can be done by writing this command:

```python
python show.py
```
