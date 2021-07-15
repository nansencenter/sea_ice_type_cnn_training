from dataclasses import dataclass
import os
from json import dump, load

import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator

from hot_encoding_utils import one_hot_m1, one_hot_m2, ice_type

class Batches:
    """
    parent class for storing the common methods of SarBatches, OutputBatches,and Amsr2Batches
    classes.
    """
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

    def convert(self, values_array, element):
        return values_array

    def resize(self, array):
        """This function resize the values of pixel of 'batches_array' with the windows of
        size 'self.step' by slicing it."""
        array = array[::self.step, ::self.step]
        if array.shape[0] % self.step:
            # in the case of image size is not being dividable to the "step" value,the value at
            # the end is omitted.
            array = array[:-1, :-1]
        return array

    def resample(self, fil, x):
        return x

    def pading(self, values_array, constant_value=0):
        """ pad based on self.pads and constant_value """
        (pad_hight_up, pad_hight_down, pad_width_west, pad_width_east) = self.pads
        values_array = np.pad(
            values_array,
            ((pad_hight_up, pad_hight_down), (pad_width_west, pad_width_east)),
            'constant',
            constant_values=(constant_value, constant_value)).astype(self.astype)
        return values_array

    def pad_and_batch(self, fil):
        """
        This function calculates the output matrix and store them in "batches_array" property of obj.
        """
        self.batches_array = {}

        for element in self.loop_list:

            values_array = np.ma.getdata(fil[self.name_for_getdata(element)])
            values_array = self.pading(values_array)
            values_array = self.resample(fil, values_array)
            views = self.view_as_windows(values_array)

            size_b, size_p, size_w1, size_w2 = views.shape
            views_2 = self.views_array(size_b, size_p, size_w1, size_w2)

            for i in range(size_b):
                for j in range(size_p):

                    view_ij = self.convert(views[i,j,:,:], element)

                    views_2[i,j,:] = view_ij.astype(self.astype)


            self.batches_array.update(
            {
                self.name_conventer(element): views_2
            }
            )

    def calculate_variable_ML(self):
        """
        This function calculates the all types of data (based on the mask) from "batches_array"
        property of archive object and store them in
        "self.PROP". Each element inside self.PROP is a list that contains slices of data for
        different locations. Each slice belongs to a specific location that the corresponding mask
        is not active (not TRUE) for that location.
        """
        PROP={}
        for element in self.loop_list:
            key = self.name_conventer(element)
            # initiation of the array
            template = []
            locs =[]
            for ix, iy in np.ndindex(self.batches_array[key].shape[:2]):
                if (~self.batches_mask[ix, iy]).all():
                    template.append(self.resize(self.batches_array[key][ix, iy]).astype(self.astype))
                    if self.__class__.__name__ == "Amsr2Batches": # only for doing it once,not several times!(only for speed)
                        locs.append((ix,iy))
            PROP.update({key: template})
            PROP.update({"_locs": locs}) if locs else None

        del self.batches_array
        del template
        return PROP

    def views_array(self, size_b, size_p, size_w1, size_w2):
        return np.zeros((size_b, size_p, size_w1, size_w2), self.astype)

    def resample(self, fil, array):
        """ Resample array on finer / lower resolution using metadata from netCDF
        Do nothing for SAR and Output
        """
        return array

class SarBatches(Batches):
    def __init__(self,archive_):
        self.loop_list = archive_.names_sar
        self.astype = np.float32
        self.pads = archive_.pads
        self.batches_mask = archive_.mask_batches
        self.window = archive_.window
        self.stride = archive_.stride
        self.step = archive_.resize_step_sar

    def resize(self, batches_array):
        """This function averages the values of pixel of 'batches_array' with the windows of
        size 'self.step' by the help of uniform_filter of scipy.
        for more information look at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html

        Next step of calculation is slicing in order to get rid of values that are belong to
        overlapping area in the filter result.
        """
        batches_array = uniform_filter(
                                        batches_array,
                                        size=(self.step,self.step),
                                        origin=(-(self.step//2),-(self.step//2))
                                        )
        return super().resize(batches_array)


class OutputBatches(SarBatches):
    def __init__(self,archive_):
        super().__init__(archive_)
        self.map_id_to_variable_values = archive_.map_id_to_variable_values
        self.names_polygon_codes = archive_.names_polygon_codes
        self.loop_list = list(range(10))
        self.astype = np.float16
        self.step = archive_.resize_step_sar

    def views_array(self, size_b, size_p, size_w1, size_w2):
        return np.zeros((size_b, size_p, size_w1, size_w2, 4), self.astype)

    def name_conventer(self, name):
        return self.names_polygon_codes[name+1]

    def name_for_getdata(self, name):
        return "polygon_icechart"

    def convert(self, values_array, element):
        return self.encode_icechart(values_array, element)

    def encode_icechart(self, values_array, element):
        """
        based on 'self.map_id_to_variable_values', all the values are converted to correct values
        of the very variable based on polygon ID values in each location in 2d array of values_array
        """
        ic = values_array.copy()
        n,p= ic.shape

        #construction of the 3D array that will be filled
        en_values_array=np.zeros((n,p,4))+np.nan

          # original dictionary should not be changed
        for id_value, variable_belong_to_id in self.map_id_to_variable_values.items():
        # each loop changes all locations of values_array (that have the very
        # 'id_value') to its corresponding value inside 'variable_belong_to_id'

            #Filling the 3D array
            en_values_array[ic == id_value,:] = np.array(variable_belong_to_id)
        return en_values_array

    def resize(self, batches_array):
        return Batches.resize(self, batches_array)


class Amsr2Batches(Batches):
    def __init__(self, archive_):
        self.loop_list = archive_.names_amsr2
        self.astype = np.float32
        self.batches_mask = archive_.mask_batches_amsr2
        self.window = archive_.window_amsr2
        self.stride = archive_.stride_amsr2

    def name_conventer(self, name):
        return name.replace(".", "_")

    def pading(self, x):
        return x

    def resample(self, fil, x):

        nc_sample = fil['sample'][:]
        nc_line = fil['line'][:]
        return x


    def resize(self, x):
        return x

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

    def __post_init__(self):
        # Each element inside self.PROP is a list that contains slices of data for
        # different locations.
        self.PROP = {}

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
            print("all files are being processed!")
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
            print(f"{filename},missing AMSR file")
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

            [ct, ca, sa, fa, cb, sb, fb, cc, sc, fc] = list(map(int, id_val_splitted[1:11]))
            #result of the one-hot encoding using method 2 with partial concentrations
            result = one_hot_m2(ct,ca,sa,fa,cb,sb,fb,cc,sc,fc)


            #Filling the dictionnary
            map_id_to_variable_values.update({int(id_val_splitted[0]): result})


        self.map_id_to_variable_values = map_id_to_variable_values

    @staticmethod
    def get_the_mask_of_sar_size_data(names_sar, fil, distance_threshold):
        mask_sar_size = np.zeros(fil[names_sar[0]][:].shape, bool)
        for str_ in names_sar+['polygon_icechart']:
            mask = np.ma.getmaskarray(fil[str_][:])
            mask_sar_size = np.ma.mask_or(mask_sar_size, mask, shrink=False)
            # not only mask itself, but also being finite is import for the data. Thus, another
            # mask also should consider and apply with 'mask_or' of numpy
            mask_isfinite = ~np.isfinite(fil[str_])
            mask_sar_size = np.ma.mask_or(mask_sar_size, mask_isfinite, shrink=False)
        # ground data is also masked
        mask_sar_size = np.ma.mask_or(mask_sar_size, np.ma.getdata(fil['distance_map']) <=
                                                                                 distance_threshold)
        return mask_sar_size

    def get_the_mask_of_amsr2_data(self, names_amsr2, fil):
        mask_amsr = np.zeros(fil[names_amsr2[0]][:].shape, bool)
        for amsr_label in names_amsr2:
            mask = np.ma.getmaskarray(fil[amsr_label])
            mask_amsr = np.ma.mask_or(mask_amsr, mask, shrink=False)
            mask_isfinite = ~np.isfinite(fil[amsr_label])
            mask_amsr = np.ma.mask_or(mask_amsr, mask_isfinite, shrink=False)
        shape_mask_amsr_0, shape_mask_amsr_1 = mask_amsr.shape[0], mask_amsr.shape[1]
        # enlarging the mask of amsr2 data to be in the size of mask sar data
        mask_amsr = np.repeat(mask_amsr, 50, axis=0)
        mask_amsr = np.repeat(mask_amsr, 50, axis=1)
        return mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1

    @staticmethod
    def pad_the_mask_of_sar_based_on_size_amsr(mask_amsr, mask_sar_size):
        # the difference between 'amsr repeated mask' and sar size mask must be padded in order to
        # centralize the scene for both sizes and having the same shape of masks
        pad_width = mask_amsr.shape[1]-mask_sar_size.shape[1]
        pad_width_west = pad_width // 2
        if (pad_width % 2) == 0:
            pad_width_east = pad_width // 2
        else:
            pad_width_east = (pad_width // 2) + 1
        pad_hight = mask_amsr.shape[0]-mask_sar_size.shape[0]
        pad_hight_up = pad_hight // 2
        if (pad_hight % 2) == 0:
            pad_hight_down = pad_hight // 2
        else:
            pad_hight_down = (pad_hight // 2) + 1
        mask_sar_size = np.pad(mask_sar_size, ((pad_hight_up, pad_hight_down),
                                               (pad_width_west, pad_width_east)),
                               'constant', constant_values=(True, True))
        pads = (pad_hight_up, pad_hight_down, pad_width_west, pad_width_east)
        return mask_sar_size, pads

    @staticmethod
    def downsample_mask_for_amsr2(final_ful_mask, shape_mask_amsr_0, shape_mask_amsr_1):
        # 4.final mask must also be available in the amsr2 size, so based on each patch of true/false
        # values inside it, it should be only one single value of true/false in that location.
        final_ful_mask_row_splitted = np.split(final_ful_mask, shape_mask_amsr_0)
        final_mask_with_amsr2_size = np.empty([shape_mask_amsr_0, shape_mask_amsr_1])
        # this loop downsize the final mask to amsr2 size with using squared patches of it
        for ii, some_rows_after_splitting in enumerate(final_ful_mask_row_splitted):
            for jj, square_shape_patch in enumerate(
                    np.hsplit(some_rows_after_splitting, shape_mask_amsr_1)):
                # below line finds out one single value of true or false
                # based on one patch of true/false values
                unique_value = np.unique(square_shape_patch)
                if unique_value.size == 2:
                    # in the case of having both False and True values inside a batch, True must
                    # be selected to indicate the very batch is masked batch.
                    unique_value = True
                final_mask_with_amsr2_size[ii, jj] = bool(unique_value)
        final_mask_with_amsr2_size = final_mask_with_amsr2_size.astype(bool)
        return final_mask_with_amsr2_size

    def calculate_mask(self, fil):
        """
        This function has four main calculation sections for calculating the mask:
        1. find out the mask of sar size data. This mask is found based on the mask of all data in
        file with sar size. combination of all masks of sar data is done with "np.ma.mask_or".
        2. find out the mask of amsr2 data. This mask is calculate by combining all the masks of
        amsr2 data in the file and the repeat element by element in order to be at the size of sar
        mask.
        3. final mask is calculated with "np.ma.mask_or" between the sar mask and the amsr2 mask.
        4. final mask should also be available for amsr2 data. Thus, it is downsized by its unique
        values in each patches of it (done in downsample_mask_for_amsr2 function).
        ====
        inputs:
        fil: netCDF4 file object
        names_amsr2: list of names of amsr2 labels in the file
        names_sar: list of names of sar labels in the file
        distance_threshold: integer indicating the threshold for considering the mask based on
        distance to land values.
        ====
        outputs:
        final_ful_mask:
        final_mask_with_amsr2_size: final_ful_mask with amsr2 shape
        pads: used pads for making two images size (coming from sar and amsr2) be at the same number
        of pixels.
        """
        # 1. get the mask of sar data
        mask_sar_size = self.get_the_mask_of_sar_size_data(self.names_sar, fil,
                                                           self.distance_threshold)
        # 2. get the mask of amsr2 data
        mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1 = self.get_the_mask_of_amsr2_data(
                                                                              self.names_amsr2, fil)
        mask_sar_size, self.pads = self.pad_the_mask_of_sar_based_on_size_amsr(mask_amsr,
                                                                               mask_sar_size)
        # 3. final mask based on two masks
        self.final_ful_mask = np.ma.mask_or(mask_sar_size, mask_amsr)  # combination of masks
        self.final_mask_with_amsr2_size = self.downsample_mask_for_amsr2(
                                           self.final_ful_mask, shape_mask_amsr_0, shape_mask_amsr_1
                                                                        )
        #if self.apply_instead_of_training:
        #    self.final_ful_mask = np.full(np.shape(self.final_ful_mask), False)
        #    self.final_mask_with_amsr2_size = np.full(np.shape(self.final_mask_with_amsr2_size),False)

    def write_batches(self):
        """
        This function writes specific slice of desired variable names (that has been stored
        previously in) self.PROP (that belongs to a specific location of scene) to a separate file.
        The file contains all variables which belongs to that location.
        """
        desired_variable_names = self.SAR_NAMES \
                                 + self.AMSR_LABELS + ['CT'] + ["_locs"]
        # removing dot from the name of variable
        desired_variable_names = [x.replace(".", "_") for x in desired_variable_names]
        # loop for saving each batch of separately in each file.
        # This way, it is compatible with the generator
        # code explained in link below
        # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        for slice_ in range(len(self.PROP['CT'])):
            # the len is equal for both sizes of input as well as the output data. Here 'CT'
            # variable is selected as one of them in the for loop.
            dict_for_saving = {}
            for name_without_dot in desired_variable_names:
                dict_for_saving.update(
                    {name_without_dot: self.PROP[name_without_dot][slice_]}
                    )
            np.savez(
                f"""{os.path.join(self.output_dir,self.scene)}_{slice_:0>6}-"""
                +f"""{self.PROP["_locs"][slice_][0]}_{self.PROP["_locs"][slice_][1]}""",
                **dict_for_saving
                )
        del dict_for_saving, self.final_ful_mask, self.final_mask_with_amsr2_size
        del self.PROP
        self.PROP = {}

    def calculate_batches_for_masks(self):
        self.mask_batches = view_as_windows(self.final_ful_mask, self.window_sar, self.stride)
        self.mask_batches_amsr2 = view_as_windows(
                    self.final_mask_with_amsr2_size, self.window_amsr2, self.stride_amsr2)

    def process_dataset(self, fil, filename):
        if self.check_file_healthiness(fil, filename):
            self.read_icechart_coding(fil, filename)
            self.calculate_mask(fil)
            self.calculate_batches_for_masks()
            for cls_ in [Amsr2Batches, SarBatches, OutputBatches, Amsr2Batches]:
                obj = cls_(self)
                obj.pad_and_batch(fil)
                self.PROP.update(obj.calculate_variable_ML())
                del obj
            del self.mask_batches_amsr2, self.mask_batches
