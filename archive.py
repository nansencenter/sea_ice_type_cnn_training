import os
from json import dump, load

import numpy as np
from skimage.util.shape import view_as_windows


class archive():
    def __init__(self, sar_names, nersc, stride_sar_size, stride_ams2_size, window_size,
                 window_size_amsr2, amsr_labels, distance_threshold, rm_swath, outpath, datapath):
        self.sar_names = sar_names
        self.nersc = nersc
        self.stride_sar_size = stride_sar_size
        self.stride_ams2_size = stride_ams2_size
        self.window_size = window_size
        self.window_size_amsr2 = window_size_amsr2
        self.amsr_labels = amsr_labels
        self.distance_threshold = distance_threshold
        self.rm_swath = rm_swath
        self.outpath = outpath
        self.datapath = datapath
        self.prop = {}

    def get_unprocessed_files(self):
        try:
            with open(os.path.join(self.outpath, "processed_files.json")) as json_file:
                self.processed_files = load(json_file)
        except FileNotFoundError:
            print("all files are being processed!")
            self.processed_files = []
        self.files = []
        for elem in os.listdir(self.datapath):
            if (elem.endswith(".nc") and elem not in self.processed_files):
                self.files.append(elem)

    def update_processed_files(self, i):

        self.processed_files.append(self.files[i]) if (
            self.files[i] and self.files[i] not in self.processed_files) else None
        with open(os.path.join(self.outpath, "processed_files.json"), 'w') as outfile:
            dump(self.processed_files, outfile)

    def check_file_healthiness(self, fil, filename):
        if 'polygon_icechart' not in fil.variables:
            print(f"'polygon_icechart' should be in 'fil.variables'. for {fil}")
            return False
        lowerbound = max([self.rm_swath, fil.aoi_upperleft_sample])
        if self.amsr_labels and not (self.amsr_labels[0] in fil.variables):
            f = open(os.path.join(self.outpath, "/discarded_files.txt"), "a")
            f.write(filename+",missing AMSR file"+"\n")
            f.close()
            print("wrote "+filename+" to discarded_files.txt in "+self.outpath)
            return False
        elif ((fil.aoi_lowerright_sample-lowerbound) < self.window_size[0] or
              (fil.aoi_lowerright_line-fil.aoi_upperleft_line) < self.window_size[1]):
            f = open(os.path.join(self.outpath, "/discarded_files.txt"), "a")
            f.write(filename+",unmasked scene is too small"+"\n")
            f.close()
            print("wrote "+filename+" to discarded_files.txt in "+self.outpath)
            return False
        else:
            return True

    def read_file_info(self, fil, filename):
        scene = filename.split('_')[0]
        # just from beginning up to variable 'FC' is considered, thus it is [:11] in line below
        names_polygon_codes = fil['polygon_codes'][0].split(";")[:11]
        map_id_to_variable_values = {}  # initialization
        # this dictionary has the ID as key and the corresponding values
        # as a list at the 'value postion' of that key in the dictionary.
        for id_and_corresponding_variable_values in fil['polygon_codes'][1:]:
            id_val_splitted = id_and_corresponding_variable_values.split(";")
            map_id_to_variable_values.update({int(id_val_splitted[0]): id_val_splitted[1:]})
        polygon_ids = np.ma.getdata(fil["polygon_icechart"])

        self.polygon_ids = polygon_ids
        self.map_id_to_variable_values = map_id_to_variable_values
        self.names_polygon_codes = names_polygon_codes
        self.scene = scene

    @staticmethod
    def get_the_mask_of_sar_data(sar_names, fil, distance_threshold):
        for str_ in sar_names+['polygon_icechart']:
            mask = np.ma.getmaskarray(fil[str_][:])
            mask = np.ma.mask_or(mask, mask)
            # not only mask itself, but also being finite is import for the data. Thus, another
            # mask also should consider and apply with 'mask_or' of numpy
            mask_isfinite = np.ma.getmaskarray(~np.isfinite(fil[str_]))
            mask = np.ma.mask_or(mask, mask_isfinite)
        # ground data is also masked
        mask_sar_size = np.ma.mask_or(mask, np.ma.getdata(fil['distance_map']) <= distance_threshold)
        return mask_sar_size

    @staticmethod
    def get_the_mask_of_amsr2_data(amsr_labels, fil):
        for amsr_label in amsr_labels:
            mask_amsr = np.ma.getmaskarray(fil[amsr_label])
            mask_amsr = np.ma.mask_or(mask_amsr, mask_amsr)
            mask_isfinite = np.ma.getmaskarray(~np.isfinite(fil[amsr_label]))
            mask_amsr = np.ma.mask_or(mask_amsr, mask_isfinite, shrink=False)
        shape_mask_amsr_0, shape_mask_amsr_1 = mask_amsr.shape[0], mask_amsr.shape[1]
        # enlarging the mask of amsr2 data to be in the size of mask sar data
        mask_amsr = np.repeat(mask_amsr, 50, axis=0)
        mask_amsr = np.repeat(mask_amsr, 50, axis=1)
        return mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1

    @staticmethod
    def padding(mask_amsr, mask_sar_size):
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
        # 4. final mask must also be available in the amsr2 size, so based on each patch of true/false
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
        amsr_labels: list of names of amsr2 labels in the file
        sar_names: list of names of sar labels in the file
        distance_threshold: integer indicating the threshold for considering the mask based on
        distance to land values.
        ====
        outputs:
        final_ful_mask:
        final_mask_with_amsr2_size: final_ful_mask with amsr2 shape
        pads: used pads for making two images size (coming from sar and amsr2) be at the same number
        of pixels
        """
        # 1. get the mask of sar data
        mask_sar_size = self.get_the_mask_of_sar_data(self.sar_names, fil, self.distance_threshold)
        # 2. get the mask of amsr2 data
        mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1 = self.get_the_mask_of_amsr2_data(
            self.amsr_labels, fil)
        mask_sar_size, self.pads = self.padding(mask_amsr, mask_sar_size)
        # 3. final mask based on two masks
        self.final_ful_mask = np.ma.mask_or(mask_sar_size, mask_amsr)  # combination of masks
        self.final_mask_with_amsr2_size = self.downsample_mask_for_amsr2(
            self.final_ful_mask, shape_mask_amsr_0, shape_mask_amsr_1)


    def create_sar_variables_for_ML_training(self, fil, mask_batches):
        """
        This function calculates the sar data and store them in a global variable with the same name
        """
        (pad_hight_up, pad_hight_down, pad_width_west, pad_width_east) = self.pads
        for sar_name in self.sar_names:
            values_array = np.ma.getdata(fil[sar_name])
            values_array = np.pad(
                values_array, ((pad_hight_up, pad_hight_down), (pad_width_west, pad_width_east)),
                'constant', constant_values=(None, None))
            batches = view_as_windows(values_array, self.window_size, self.stride_sar_size)
            # initiation of the array with one single layer of empty data (meaningless values)
            template_sar_float32 = np.empty(self.window_size).astype(np.float32)
            for ix, iy in np.ndindex(batches.shape[:2]):
                if (~mask_batches[ix, iy]).all():
                    # stack data in 3rd dimension
                    template_sar_float32 = np.dstack((template_sar_float32, batches[ix, iy]))
            # remove the first empty values
            self.prop.update({sar_name: template_sar_float32[:, :, 1:].astype(np.float32)})
            del template_sar_float32
            del values_array
            del batches

    def create_output_variables_for_ML_training(self, mask_batches):
        """
        This function calculates the output data and store them in a global variable with the same name
        of them in the file for example 'CT' or 'CA' etc.
        """
        (pad_hight_up, pad_hight_down, pad_width_west, pad_width_east) = self.pads
        for index in range(10):  # iterating over variables that must be saved for example 'CT' or 'CA'
            values_array = self.polygon_ids
            values_array = np.pad(
                values_array, ((pad_hight_up, pad_hight_down),
                               (pad_width_west, pad_width_east)),
                'constant', constant_values=(0, 0)).astype(np.byte)
            batches = view_as_windows(values_array, self.window_size, self.stride_sar_size)
            # initiation of the array with one single layer of empty data (meaningless values)
            template_sar = np.empty(self.window_size).astype(np.byte)
            for ix, iy in np.ndindex(batches.shape[:2]):
                if (~mask_batches[ix, iy]).all():
                    raw_id_values = batches[ix, iy]
                    for id_value, variable_belong_to_id in self.map_id_to_variable_values.items():
                        # each loop changes all locations of raw_id_values (that have the very
                        # 'id_value') to its corresponding value inside 'variable_belong_to_id'
                        raw_id_values[raw_id_values == id_value] = np.byte(
                            variable_belong_to_id[index])
                    # stack to the data in 3rd dimension
                    template_sar = np.dstack((template_sar, raw_id_values))

            self.prop.update({self.names_polygon_codes[index+1]: template_sar[:, :, 1:]})
            del template_sar
            del batches
            del values_array

    def create_amsr2_variables_for_ML_training(self, fil):
        """
        This function calculates the amsr2 data and store them in a global variable with the same name
        """
        mask_batches_amsr2 = view_as_windows(
            self.final_mask_with_amsr2_size, self.window_size_amsr2, self.stride_ams2_size)
        for amsr_label in self.amsr_labels:
            values_array = np.ma.getdata(fil[amsr_label])
            batches = view_as_windows(values_array, self.window_size_amsr2, self.stride_ams2_size)
            # initiation of the array with one single layer of data
            template_amsr2 = np.empty(self.window_size_amsr2).astype(np.float32)
            for ix, iy in np.ndindex(batches.shape[:2]):
                if (~mask_batches_amsr2[ix, iy]).all():
                    # stack the data in 3rd dimension
                    template_amsr2 = np.dstack((template_amsr2, batches[ix, iy]))
            self.prop.update({amsr_label.replace(".", "_"): template_amsr2[:, :, 1:].astype(np.float32)})
            del template_amsr2
            del batches
            del values_array
        del mask_batches_amsr2

    def write_scene_files(self):
        desired_variable_names = self.sar_names+self.amsr_labels+self.names_polygon_codes[1:]
        # removing dot from the name of variable
        desired_variable_names = [x.replace(".", "_") for x in desired_variable_names]
        # loop for saving each batch of separately in each file.
        # This way, it is compatible with the generator
        # code explained in link below
        # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        for i in np.arange(np.shape(self.prop['CT'])[2]):
            # third dim is equal for both sizes of input as well as the output data.Here CT
            # variable is selected as one of them in the for loop.
            dict_for_saving = {}
            for name_without_dot in desired_variable_names:
                dict_for_saving.update(
                    {name_without_dot: self.prop[name_without_dot][:, :, i]})
            np.savez(f"{os.path.join(self.outpath,self.scene)}_{i:0>6}_{self.nersc}", **dict_for_saving)
        del dict_for_saving, self.final_ful_mask, self.final_mask_with_amsr2_size
        del self.prop
        self.prop = {}
