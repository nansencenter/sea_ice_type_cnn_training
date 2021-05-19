# description
This repository provides the facilities for ASIP v2 data [(webpage)](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134). Manual of this dataset is provided [here](https://data.dtu.dk/ndownloader/files/24951176).
### DISCLAIMER: This code project is released as it without guarentees and extensive testing. It is meant to guide and help researchers and students get started on sea ice modelling with convolutional neural networks.
The order of execution of different parts of the code is as follow:
 1. [Execute the data building](#requirement)
 2. [Execute the tensorflow training](#execute-the-data-building)
 3. [Execute the inference (apply) code](#execute-the-inference-code)
 4. [Plotting the result of inference](#plotting-the-result-of-inference)

# Requirement
just run the following command in your environment in order to install the requirements:

`pip install -r requirements.txt`
The users of Microsoft VScode can easily open the remote development container with the help of `.devcontainer` folder and Dockerfile
# Usage
This code uses [python argparse](https://docs.python.org/3/library/argparse.html) which the details of developed arguments are shown in the table below. They can be given to the command above as an argument of command:

| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  []                 |  []                 |  [no default]| The first and the only positional argument is the path to directory with input netCDF files needed for data building or applying the trained model (inference)
|  -o                 |  --output_dir       | same path of `build_dataset.py` file      |Path to directory with output files
|  -n                 |    --noise_method   |'nersc_'|the method that error calculation had  been used for error. Leave as empty string '' for ESA noise corrections or as 'nersc_' for the Nansen center noise correction.
|  -w                 |   --window_size     |  700 | window size (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the size of image samples that has been used for ML training step)
|  -s                 |   --stride          |  700 | stride (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the stride that determines the overlapping areas between image samples for ML training step)
|  -r                 |   --aspect_ratio     |  [] | The ration between the cell size of primary and secondary input of ML model. stride and window_size must be dividable to it.
|  -i                 |   --apply_instead_of_training  |  False (in the case of absence in the arguments) | Flag for distinguishing the two mode of data building (training or inference). This flag is working based on its ABSENCE or PRESENCE. In the PRESENCE case, the code will consider all locations of the scene for inference purposes of the scene (not for training).
|  -swa                 |   --rm_swath        |  0    |threshold value for comparison with netCDF file.aoi_upperleft_sample to border the calculation
|  -d                 |   --distance_threshold |  0  |threshold for distance from land in mask calculation
|  -a                 |   --step_resolution_sar | 1  |step for resizing the sar data (default value leads to no resizing)
|  -b                 |   --step_resolution_output|1 |step for resizing the ice chart data (default value leads to no resizing)
|  -see                |   --shuffle_on_epoch_end  |  False (in the case of absence in the arguments) | Flag for Shuffling the training subset of IDs at the end of every epoch during the training
|  -sft                |   --shuffle_for_training  |  False (in the case of absence in the arguments) | Flag for Shuffling the list of IDs before dividing it into two 'training' and 'validation' subsets
|  -m                |   --memory_mode  |  False (in the case of absence in the arguments) | Flag for use memory instead of npz files for the input of inference of the scene (not for training). It is only available for applying the model (inference mode).
|  -bd                |   --beginning_day_of_year          |  0 | min threshold value for comparison with scenedate of files for considering a limited subset of files based on their counts from the first of january of the same year
|  -ed                |   --ending_day_of_year          |  365 | max threshold value for comparison with scenedate of files for considering a limited subset of files based on their counts from the first of january of the same year
|  -p                |   --precentage_of_training          |  [] | percentage of IDs that should be considered as training data (between 0,1). '1-precentage_of_training' fraction of data is considered as validation data. NOT APPLICABLE for building the data into npz files.
|  -bs               |   --batch_size          |  [] | batch size for data generator


# Execute the data building
By just giving the full absolute address of the folder that contains all of the uncompressed .nc files of ASIP data, data building part of code is able to build the data based on those files and make them ready for further Machine learning training or inference activities. This command is executable with two different modes, name Training and Inference which are determined by ABSENCE or PRESENCE of `-i` in the command line.

This can be done by writing this command for training purposes:

```python
python build_dataset.py /absolute/path/to/the/folder/of/input_files
```

And also can be done by writing this command for building data in order to apply the trained model to some data (inference purposes):

```python
python build_dataset.py /absolute/path/to/the/folder/of/input_files -i
```

In the first case, only the unmasked locations of data are selected among the others for having a completely clean data in order to feed it for ML training. In the second case, data of all locations (regardless of their masks) are built into npz files that can be used for reconstructing the image from the output of the model.

This command will create a folder named **output** in the folder that contains the `build_dataset.py` and write all the output files into it.

>  HINT: The folder containing input files must have all of the **.nc** files without any subfoldering

As an example, for the case of building data from **/fold1** folder and store them in **/fold2** folder with nersc noise calculation, having window size and stride of **400**,and execution for training mode (not for inference) the command below is used:

```python
python build_dataset.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400 -r 50
```
The above command can be executed by adding **-i** in the argument for building data for inference purposes.

# Execute the tensorflow training
After building the data, you can train the tensorflow model with those `.npz` files as the result of
data building calculation. To do this, run the script `apply_or_train_model.py` by setting the address of output folder (which has been used with training mode) from pervious calculation (data building) with '-o' in the arguments.

> It is strongly recommend to read the link below before using this part of the code because everything for file based config (including the classes and scripts) is developed based on explanation of this web page:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


If you want to run the training with scenes that are belong to a specific season of the year(spring,summer,etc), then you can set `beginning_day_of_year` and `ending_day_of_year` variable in the arguments of commandline in order to make use of the files that are only belong to the period of year between these two numbers. These two numbers are start and end day count from the beginning of the year for reading data between them.


Train the tensorflow can be done by writing this command in order to training from npz files of *fold1* folder:

```python
python apply_or_train_model.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400 -r 50 -bs 4 -p 0.8 -see -sft
```
In the above example **/fold1** is not applicable and the npz files are being read from /fold2 folder.

# Execute the inference code
For seeing the result of network after training, `apply_or_train_model.py` can be used. To do this, just like previous example of commandline of training, we can run the same command with additional *-i* argument and *p=1.0* in the commandline:
```python
python apply_or_train_model.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400 -r 50 -bs 4 -p 1.0 -i
```
Place the shuffling flags like -see and -sft in incorrect for inference mode. All the data must be used for applying model (no validation data), so percentage of training values **must** be equal to 1.0 for it.

This mode can also be executed in memory based manner. In this case, only the 'nc' files of /fold1 is taking into consideration for applying the trained model on them.

As an example, the memory-based execution of inference mode of above command would be:
```python
python apply_or_train_model.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400 -r 50 -bs 4 -p 1.0 -i -m
```

A folder named 'reconstructs_folder' will be created at the same level of input directory (for memory-based manner) or one level up in foldering hierarchy (for file-based manner) in order not to put the reconstructed ones and npz files in the same folder.

> **Hint**: If you use resizing for building the data and then train the network with the resized data, this inference code (and consequent plotting) is not applicable.
**This inference code is only for cases that resizing is not used.**


# Plotting the result of inference


For plotting, unlike all pervious executions, you need to run it from outside the development container of VScode. It means you have to install scipy and numpy on your env and run the `show.py` with the python interpreter outside the container. This code can also be substituted with an interactive jupyter-notebook.
Plotting can be done by writing this command:

```python
python show.py
```
