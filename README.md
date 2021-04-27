# description
This data builder part of this repo (`data_builder.py`) provides the clean version of ASID v2 data [(webpage)](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134). Manual of this dataset is provided [here](https://data.dtu.dk/ndownloader/files/24951176).
### DISCLAIMER: This code project is released as it without guarentees and extensive testing. It is meant to guide and help researchers and students get started on sea ice modelling with convolutional neural networks.
# Requirement
just run the following command in your environment in order to install the requirements:

`pip install -r requirements.txt`

The users of Microsoft VScode can easily open the remote development container with the help of `.devcontainer` folder and Dockerfile
# Execut the data building
By just giving the full absolute address of the folder that contains all of the uncompressed .nc files of ASID data, this code is able to bluid the data based on those files and make them ready for further Machine learning training activities.

This can be done writing this command:

`python data_builder.py /absolute/path/to/the/folder/of/input/files`

This command will create a folder named **output** in the folder that contains the `data_builder.py` and write all the output files into it.

>  HINT: The folder containing input files must have all of the **.nc** files without any subfoldering

The output address as well as some other parameters for this process is configurable. They can be given to the command above as an agrument of command. Table below shows the detail of them:

| Argument short form | Argument long form  | default value | Description
| ------------------- | --------------------|-------------- | --------------
|  []                 |  []                 |  [no default]| The first and the only positional argument is the path to directory with input netCDF files needed for data building
|  -o                 |  --output_dir       | same path of `data_builder.py` file      |Path to directory with output files
|  -n                 |    --noise_method   |'nersc_'|the method that error calculation had  been used for error. Leave as empty string '' for ESA noise corrections or as 'nersc_' for the Nansen center noise correction.
|  -w                 |   --window_size     |  700 | window size (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the size of image samples that has been used for ML training step)
|  -s                 |   --stride          |  700 | stride (of sar and ice chart data) for batching calculation (must be dividable to 50)(This will be the stride that determines the overlapping areas between image samples for ML training step)
|  -r                 |   --rm_swath        |  0    |threshold value for comparison with netCDF file.aoi_upperleft_sample to border the calculation
|  -d                 |   --distance_threshold |  0  |threshold for distance from land in mask calculation
|  -a                 |   --step_resolution_sar | 1  |step for resizing the sar data (default value leads to no resizing)
|  -b                 |   --step_resolution_output|1 |step for resizing the ice chart data (default value leads to no resizing)


As an example for the case of building data from **/fold1** folder and store them in **/fold2** folder with nersc noise calculation and having window size and stride of **400**, the command below is used:

`python data_builder.py /fold1 -o /fold2 -n nersc_ -w 400 -s 400`


# Execute the tensorflow training
After building the data, you can train the tensorflow model with those `.npz` files as the result of
data building calculation. To do this, run the script `keras_script.py` by setting the address of output folder from pervious calcultion (data building) to the `mypath` variable in the script.

> It is strongly recommend to read the link below before using this part of the code because everything (including the classes and script) is developed based on explantion of this web page:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


If you want to run the training with scenes that are belong to a specific season of the year(spring,summer,etc), then you can set `beginning_day_of_year` and `ending_day_of_year` variable in the script in order to make use of the files that are only belong to the period of year between these two numbers. These two numbers are start and end day count from the begning of the year for reading data between them.
