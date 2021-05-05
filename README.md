# description
This repository provides the facilities for ASIP v2 data [(webpage)](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134). Manual of this dataset is provided [here](https://data.dtu.dk/ndownloader/files/24951176).
### DISCLAIMER: This code project is released as it without guarentees and extensive testing. It is meant to guide and help researchers and students get started on sea ice modelling with convolutional neural networks.
The order of execution of different parts of the code is as follow:
 1. [Execute the data building](#requirement)
 2. [Execute the tensorflow training](#execute-the-data-building)
 3. [Execute the inference code](#execute-the-inference-code)
 4. [Plotting the result of inference](#plotting-the-result-of-inference)

# Requirement
just run the following command in your environment in order to install the requirements:

`pip install -r requirements.txt`

The users of Microsoft VScode can easily open the remote development container with the help of `.devcontainer` folder and Dockerfile
# Execute the data building
By just giving the full absolute address of the folder that contains all of the uncompressed .nc files of ASIP data, data building part of code is able to build the data based on those files and make them ready for further Machine learning training activities.

This can be done writing this command:

```python
python build_dataset.py /absolute/path/to/the/folder/of/input_files
```

This command will create a folder named **output** in the folder that contains the `build_dataset.py` and write all the output files into it. This command is executable with two different modes, name Training and Inference which are determined by ABSENCE or PRESENCE of `-i` in the commandline.

>  HINT: The folder containing input files must have all of the **.nc** files without any subfoldering

The output address as well as some other parameters for this process is configurable. They can be given to the command above as an argument of command. Table below shows the detail of them:

| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  []                 |  []                 |  [no default]| The first and the only positional argument is the path to directory with input netCDF files needed for data building
|  -o                 |  --output_dir       | same path of `build_dataset.py` file      |Path to directory with output files
|  -n                 |    --noise_method   |'nersc_'|the method that error calculation had  been used for error. Leave as empty string '' for ESA noise corrections or as 'nersc_' for the Nansen center noise correction.
|  -w                 |   --window_size     |  700 | window size (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the size of image samples that has been used for ML training step)
|  -i                 |   --inference_mode  |  False (in the case of absence in the arguments) | Flag for distinguishing the two mode of data building (training or inference). This flag is working based on its ABSENCE or PRESENCE.
|  -s                 |   --stride          |  700 | stride (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the stride that determines the overlapping areas between image samples for ML training step)
|  -r                 |   --rm_swath        |  0    |threshold value for comparison with netCDF file.aoi_upperleft_sample to border the calculation
|  -d                 |   --distance_threshold |  0  |threshold for distance from land in mask calculation
|  -a                 |   --step_resolution_sar | 1  |step for resizing the sar data (default value leads to no resizing)
|  -b                 |   --step_resolution_output|1 |step for resizing the ice chart data (default value leads to no resizing)


As an example, for the case of building data from **/fold1** folder and store them in **/fold2** folder with nersc noise calculation, having window size and stride of **400**,and execution for training mode (not for inference) the command below is used:

```python
python build_dataset.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400
```


# Execute the tensorflow training
After building the data, you can train the tensorflow model with those `.npz` files as the result of
data building calculation. To do this, run the script `keras_script.py` by setting the address of output folder (which has been used with training mode) from pervious calculation (data building) to the `outputpath` variable in the script.

> It is strongly recommend to read the link below before using this part of the code because everything (including the classes and script) is developed based on explanation of this web page:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


If you want to run the training with scenes that are belong to a specific season of the year(spring,summer,etc), then you can set `beginning_day_of_year` and `ending_day_of_year` variable in the script in order to make use of the files that are only belong to the period of year between these two numbers. These two numbers are start and end day count from the beginning of the year for reading data between them.


Train the tensorflow can be done writing this command:

```python
python train_model.py
```
# Execute the inference code
For seeing the result of network after training, `apply_model.py` can be used. To do this, just set the **stride** and **window size** in variables `stride` and `ws` equal to the values used for building the data. In this script, `outputpath` is the folder path of output of data building calculation (with inference mode activated by putting `-i` in arguments) and `netcdfpath` is the path of netcdf files that are being read for data building (as the input of data building). This script will create a folder named `reconstructs_folder` in the same folder that contains `output` folder and write its results in it.
> **Hint**: If you use resizing for building the data and then train the network with the resized data, this inference code (and consequent plotting) is not applicable.
**This inference code is only for cases that resizing is not used.**

Execute the inference code can be done writing this command:

```python
python apply_model.py
```

# Plotting the result of inference
For plotting, unlike all pervious executions, you need to run it from outside the development container of VScode. It means you have to install scipy and numpy on your env and run the `show.py` with the python interpreter outside the container. This code can also be substituted with an interactive jupyter-notebook.
Plotting can be done writing this command:

```python
python show.py
```
