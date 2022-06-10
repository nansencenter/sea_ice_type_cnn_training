from dataclasses import dataclass
import os
from json import dump, load
import time

import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


class Batches:
    """
    parent class for storing the common methods of SarBatches, OutputBatches,and Amsr2Batches
    classes.
    """
    def get_array(self, fil, name):
        return fil[name][:].astype(self.astype).filled(np.nan)

    def convert(self, array):
        return array

    def view_as_windows(self, array):
        window_size = (self.window, self.window)
        stride = self.stride
        if len(array.shape)==3:
            window_size += (array.shape[2],)
            stride = (self.stride, self.stride, 1)
        return view_as_windows(array, window_size, stride)

    def name_conventer(self, name):
        return name

    def name_for_getdata(self, name):
        return name

    def resize(self, array):
        """This function resize the values of pixel of 'batches_array' with the windows of
        size 'self.step' by slicing it."""
        array = array[::self.step, ::self.step]
        if array.shape[0] % self.step:
            # in the case of image size is not being dividable to the "step" value,the value at
            # the end is omitted.
            array = array[:-1, :-1]
        return array

    def make_batch(self, fil):
        """
        This function calculates the output matrix and store them in "batches_array" property of obj.
        """
        batch = {}
        for element in self.loop_list:
            array = self.get_array(fil, self.name_for_getdata(element))
            views = self.view_as_windows(array)
            array_list = []
            array_locs = []
            for i in range(views.shape[0]):
                for j in range(views.shape[1]):
                    if (self.check_view(views[i,j,:,:])):
                        continue
                    array_list.append(self.resize(self.convert(views[i,j,:,:])))
                    array_locs.append((i,j))
            batch[self.name_conventer(element)] = array_list
            batch[self.name_conventer(element) + '_loc'] = array_locs
        return batch

    def resample(self, fil, array):
        """ Resample array on finer / lower resolution using metadata from netCDF
        Do nothing for SAR and Output
        """
        return array
    
    def check_view(self,view):
        """
        to know if there is a nan in the view
        """
        bol=False
        if np.any(np.isnan(view[:,:])):
            bol=True
        return bol


class SarBatches(Batches):
    def __init__(self,archive_):
        self.loop_list = archive_.names_sar
        self.astype = np.float32
        self.window = archive_.window_sar
        self.stride = archive_.stride_sar
        self.step = archive_.resize_step_sar

    def resize(self, batches_array):
        """This function averages the values of pixel of 'batches_array' with the windows of
        size 'self.step' by the help of uniform_filter of scipy.
        for more information look at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html

        Next step of calculation is slicing in order to get rid of values that are belong to
        overlapping area in the filter result.
        """
        if self.step != 1:
            batches_array = uniform_filter(
                                            batches_array,
                                            size=(self.step,self.step),
                                            origin=(-(self.step//2),-(self.step//2))
                                            )
            batches_array = super().resize(batches_array)
        return batches_array


class OutputBatches(SarBatches):
    def __init__(self, archive_):
        super().__init__(archive_)
        self.map_id_to_variable_values = archive_.map_id_to_variable_values
        self.names_polygon_codes = archive_.names_polygon_codes
        self.loop_list = [0]
        self.astype = np.float16
        self.window = archive_.window_sar
        self.stride = archive_.stride_sar
        self.step = archive_.resize_step_sar

    def name_conventer(self, name):
        return 'ice_type'

    def name_for_getdata(self, name):
        return "polygon_icechart"

    def convert(self, array):
        """
        based on 'self.map_id_to_variable_values', all the values are converted to correct values
        of the very variable based on polygon ID values in each location in 2d array of values_array
        return only one vector because all pixels in the view are in the same segment so the same information
        """
        for id_value, variable_belong_to_id in self.map_id_to_variable_values.items():
            if id_value==array[0][0]:
                result=variable_belong_to_id
        return result

    def resize(self, array):
        return array
    
    def check_view(self,view):
        """
        to know if there is a nan in the view and if there are several different segments in the view
        """
        bol=False
        if np.any(np.isnan(view[:,:])):
            bol=True
        if(np.amax(view[:,:]) != np.amin(view[:,:])):
            bol=True
        return bol


class DistanceBatches(Batches):
    def __init__(self, archive_):
        self.map_id_to_variable_values = archive_.map_id_to_variable_values
        self.names_polygon_codes = archive_.names_polygon_codes
        self.loop_list = [0]
        self.astype = np.float16
        self.window = archive_.window_sar
        self.stride = archive_.stride_sar
        self.step = archive_.resize_step_sar

    def get_array(self, fil, name):
        """
        create the distance matrix between the pixels and the borders of the icechart polygons 
        """
        poly=fil[name][:].astype(self.astype).filled(np.nan)
        list_poly=np.unique(poly)
        distance=distance_transform_edt(poly==list_poly[0], return_distances=True, return_indices=False)
        for id_poly in list_poly[1:] :
            distance1=distance_transform_edt(poly==id_poly, return_distances=True, return_indices=False)
            distance[distance == 0] = distance1[distance == 0]
        return distance

    def name_for_getdata(self, name):
        return "polygon_icechart"

    def convert(self, array):
        """
        return only the value of the distance for the middle pixel of the view
        """
        n,p=array.shape
        return array[n//2][p//2]

    def resize(self, array):
        return array

    def name_conventer(self, name):
        return 'distance_border'

    def check_view(self,view):
        """
        to know if there is a nan in the view
        """
        bol=False
        if (view[:,:].size - np.count_nonzero(view[:,:])!=0):
            bol=True
        return bol


class Amsr2Batches(Batches):
    def __init__(self, archive_):
        self._archive = archive_
        self.loop_list = archive_.names_amsr2
        self.astype = np.float32
        self.window = archive_.window_amsr2
        self.stride = archive_.stride_amsr2
        self.resample_step = archive_.resample_step_amsr2

    def name_conventer(self, name):
        return name.replace(".", "_")

    def resize(self, x):
        return x

    def get_array(self, fil, name):
        return self._archive.amsr2_data[name].astype(self.astype)


@dataclass
class Archive():
    input_dir           : str
    output_dir          : str
    names_sar           : list
    names_amsr2         : list
    window_sar          : int
    window_amsr2        : int
    stride_sar          : int
    stride_amsr2        : int
    resample_step_amsr2 : int
    resize_step_sar     : int
    rm_swath            : int
    distance_threshold  : int
    encoding            : str

    def get_unprocessed_files(self):
        """
        Two function do two jobs:
        1. Read the list of processed files from 'processed_files.json'
        2. find out which files in directory of archive has not been processed compared to
        'self.processed_files'  and save them as 'self.files'. """
        try:
            with open(os.path.join(self.output_dir, "processed_files.json")) as json_file:
                self.processed_files = load(json_file)
        except FileNotFoundError:
            print("All files are being processed!")
            self.processed_files = []
        self.files = []
        for elem in os.listdir(self.input_dir):
            if (elem.endswith(".nc") and elem not in self.processed_files):
                self.files.append(elem)

    def update_processed_files(self, i):
        """update 'self.processed_files' based on 'self.files' and store the with a file named
        'processed_files.json'. """
        self.processed_files.append(self.files[i]) if (
            self.files[i] and self.files[i] not in self.processed_files) else None
        with open(os.path.join(self.output_dir, "processed_files.json"), 'w') as outfile:
            dump(self.processed_files, outfile)

    def check_file_healthiness(self, fil, filename):
        """Check the healthiness of file by checking the existence of 'polygon_icechart' and
        AMSR LABELS in the 'variables' section of NETCDF file. The comparison of window size and
        size of the file is also done at the end. """
        if 'polygon_icechart' not in fil.variables:
            print(f"'polygon_icechart' should be in 'fil.variables'. for {filename}")
            return False
        if not (self.names_amsr2[0] in fil.variables):
            print(f"{filename}, missing AMSR file")
            return False
        lowerbound = max([self.rm_swath, fil.aoi_upperleft_sample])
        if ((fil.aoi_lowerright_sample-lowerbound) < self.window_sar or
                (fil.aoi_lowerright_line-fil.aoi_upperleft_line) < self.window_sar):
            print(f"{filename},unmasked scene is too small")
            return False
        else:
            return True

    def read_icechart_coding(self, fil, filename):
        """
        based on 'polygon_codes' and 'polygon_icechart' section of netCDF file as well as the name
        of the file, this function set the values of properties of 'polygon_ids', 'scene',
        'names_polygon_codes' and 'map_id_to_variable_values' of archive object.
        """
        self.scene = filename.split('_')[0]
        # just from beginning up to variable 'FC' is considered, thus it is [:11] in line below
        self.names_polygon_codes = fil['polygon_codes'][0].split(";")[:11]
        self.polygon_ids = np.ma.getdata(fil["polygon_icechart"])
        map_id_to_variable_values = {}  # initialization

        # this dictionary has the ID as key and the corresponding values
        # as a list at the 'value postion' of that key in the dictionary.
        for id_and_corresponding_variable_values in fil['polygon_codes'][1:]:
            id_val_splitted = id_and_corresponding_variable_values.split(";")
            result = list(map(int, id_val_splitted[1:11]))
            #Filling the dictionnary
            map_id_to_variable_values.update({int(id_val_splitted[0]): result})
        self.map_id_to_variable_values = map_id_to_variable_values

    def resample_amsr2(self, fil):
        """ Crop AMSR2 data to SAR extent and resample to desired resolution """
        line = fil['line'][:]
        sample = fil['sample'][:]
        sar_shape = fil['sar_primary'].shape
        self.line = np.arange(int(self.resample_step_amsr2/2), sar_shape[0], self.resample_step_amsr2)
        self.sample = np.arange(int(self.resample_step_amsr2/2), sar_shape[1], self.resample_step_amsr2)
        sample_grid, line_grid = np.meshgrid(self.sample, self.line)
        self.amsr2_data = {}
        for name in self.names_amsr2:
            x = fil[name][:].filled(np.nan)
            rgi = RegularGridInterpolator((line, sample), x, bounds_error=False, fill_value=None)
            self.amsr2_data[name] = rgi((line_grid, sample_grid))

    def write_batches(self):
        """
        This function writes specific slice of desired variable names (that has been stored
        previously in) self.PROP (that belongs to a specific location of scene) to a separate file.
        The file contains all variables which belongs to that location.
        """
        inp_var_names = [x for x in self.batches.keys() if not x.endswith('_loc')]
        out_var_names = [x.replace('.', '_') for x in inp_var_names]
        loc_names = [x for x in self.batches.keys() if x.endswith('_loc')]

        for i, loc in enumerate(self.batches[loc_names[0]]):
            # check that current loc is present in all locs
            loc_exists = True
            for loc_name in loc_names:
                if loc not in self.batches[loc_name]:
                    loc_exists = False
                    break
            # skip if loc is not present in all
            if not loc_exists:
                continue
            data = {}
            for iname, oname, lname in zip(inp_var_names, out_var_names, loc_names):
                j = self.batches[lname].index(loc)
                data[oname] = self.batches[iname][j]
            opath = os.path.join(self.output_dir, self.scene)
            ofilename = f'{opath}_{i:0>6}.npz'
            np.savez(ofilename, **data)

    def process_dataset(self, fil, filename):
        t0 = time.time()
        if self.check_file_healthiness(fil, filename):
            self.read_icechart_coding(fil, filename)
            self.resample_amsr2(fil)
            for cls_ in [SarBatches, OutputBatches, DistanceBatches, Amsr2Batches]:
                obj = cls_(self)
                batch = obj.make_batch(fil)
                self.batches.update(batch)
                del obj
