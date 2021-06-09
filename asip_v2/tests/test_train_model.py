import sys
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
from train_model import FileBasedConfigure, read_input_params_for_training


class FileBasedConfigureTestCases(unittest.TestCase):
    """ Tests for FileBasedConfigure"""
    @mock.patch('train_model.np.load', return_value={'nersc_sar_primary':np.zeros([10, 20]),
                                                     'CT':np.zeros([30, 40]),
                                                     'btemp_6_9h':np.zeros([50, 60])})
    @mock.patch('train_model.Archive.__init__', return_value=None)
    def test_function_calculate_dims(self, mock_archive, mock_np_load):
        """ shall set the correct dims """
        config_ = FileBasedConfigure(archive=mock_archive)
        config_.id_list = [""]
        config_.calculate_dims()
        self.assertEqual(config_.dims_input, (10, 20))
        self.assertEqual(config_.dims_output, (30, 40))
        self.assertEqual(config_.dims_amsr2, (50, 60))

    @mock.patch('train_model.Archive.__init__', return_value=None)
    def test_function_filling_id_list(self, mock_archive):
        """ shall set the correct npz files based on the time """
        config_ = FileBasedConfigure(archive=mock_archive)
        config_.list_of_names = ['/workspaces/ASIP-v2-builder/out/20180410T084537_00-0_0.npz',
                                 '/workspaces/ASIP-v2-builder/out/20190410T084537_01-0_1.npz',
                                 '/workspaces/ASIP-v2-builder/out/20190510T042318_00-0_2.npz']
        # period of time is considered as the 4th month of each year
        config_.BEGINNING_DAY_OF_YEAR = 90
        config_.ENDING_DAY_OF_YEAR = 120
        config_.filling_id_list()
        # The one that belongs to 5th month of year shoud not be in the 'id_list'
        self.assertEqual(config_.id_list,
                         ['/workspaces/ASIP-v2-builder/out/20180410T084537_00-0_0.npz',
                          '/workspaces/ASIP-v2-builder/out/20190410T084537_01-0_1.npz'])

    @mock.patch('train_model.Archive.__init__', return_value=None)
    def test_function_instantiate_image_with_zeros_and_get_the_patch_locations_of_image(self,
                                                                                      mock_archive):
        """ shall instantiate the image array with proper size and set the patch locations"""
        config_ = FileBasedConfigure(archive=mock_archive)
        config_.list_of_names = ['/workspaces/ASIP-v2-builder/out/20180410T084537_00-0_0.npz',
                                 '/workspaces/ASIP-v2-builder/out/20180410T084537_01-0_1.npz',
                                 '/workspaces/ASIP-v2-builder/out/20180410T084537_02-5_9.npz']

        config_.WINDOW_SIZE = (2, 3)
        config_.instantiate_image_with_zeros_and_get_the_patch_locations_of_image()
        self.assertEqual(config_.patch_locs, [(0, 0), (0, 1), (5, 9)])
        # The result must be (5+1)*2, (9+1)*3 for below line:
        self.assertEqual(config_.img.shape, (12, 30))

    @mock.patch('train_model.Archive.__init__', return_value=None)
    def test_function_read_input_params_for_training(self, mock_archive):
        """input from sys.argv should be read correctly"""
        sys.argv = ['','-o', 'dir_name', '-see', '-bd', '7', '-ed', '90', '-p', '.7', '-bs','4']
        read_input_params_for_training()
        mock_archive.assert_called_with(
                            apply_instead_of_training=False, batch_size=4, beginning_day_of_year=7,
                            ending_day_of_year=90, outpath='dir_name', percentage_of_training=0.7,
                            shuffle_for_training=False, shuffle_on_epoch_end=True
                                       )
