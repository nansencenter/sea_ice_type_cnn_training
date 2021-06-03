import unittest
import unittest.mock as mock

import numpy as np
from apply_model import MemoryBasedConfigure
from data_generator import (DataGenerator, DataGeneratorFrom_npz_File,
                            DataGeneratorFromMemory)
from train_model import FileBasedConfigure
from utility import Archive, Configure


class DataGeneratorTestCases(unittest.TestCase):

    def setUp(self):
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.dims_input = (700, 700)
        config_.dims_output = (700, 700)
        config_.dims_amsr2 = (14, 14)
        config_.input_var_names = ["input_var_name1", "input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1", "amsr2_var_names2"]
        config_.batch_size = 2
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [56, 87, 34, 12, 90, 75, 88]}# a list with length of seven
        config_.set_params()
        self.test_generator = DataGenerator(config_.partition['train'], **config_.params)

    def tearDown(self):
        del self.test_generator

    def test_len(self):
        self.assertEqual(len(self.test_generator), 3) # 7//2=3

    def test_function_x_y_z_initialization(self):
        self.test_generator.x_y_z_initialization()
        self.assertEqual(self.test_generator.y.shape, (2, 700, 700, 1))
        self.assertEqual(self.test_generator.X.shape, (2, 700, 700, 2))
        self.assertEqual(self.test_generator.z.shape, (2, 14, 14, 2))

    def test_function_on_epoch_end(self):
        self.test_generator.on_epoch_end()
        # regardless of values inside the 'train' list, the indexes must be the array containing
        # the numbers up to len of it
        np.testing.assert_equal(self.test_generator.indexes, np.array([0, 1, 2, 3, 4, 5, 6]))

class DataGeneratorFrom_npz_FileTestCases(unittest.TestCase):
    """Tests for DataGeneratorFrom_npz_File"""

    # The first six row of side_effect belongs to the first batch of data, and the second six rows
    # belong to the second batch of data. Since we want to have different response of file loading
    # for this mock, we need to use this way of side effect to resemble file loading activities.
    # Since the np.load is called five times for each of i in self.list_IDs_temp, the side effect must
    # have the effect five times.
    @mock.patch('data_generator.np.load', side_effect=[
    {"nersc_sar_primary":(3,), "nersc_sar_secondary":(5,),"CT":(7,), "btemp_6_9h":(9), "btemp_6_9v":(11,)},
    {"nersc_sar_primary":(3,), "nersc_sar_secondary":(5,),"CT":(7,), "btemp_6_9h":(9), "btemp_6_9v":(11,)},
    {"nersc_sar_primary":(3,), "nersc_sar_secondary":(5,),"CT":(7,), "btemp_6_9h":(9), "btemp_6_9v":(11,)},
    {"nersc_sar_primary":(3,), "nersc_sar_secondary":(5,),"CT":(7,), "btemp_6_9h":(9), "btemp_6_9v":(11,)},
    {"nersc_sar_primary":(3,), "nersc_sar_secondary":(5,),"CT":(7,), "btemp_6_9h":(9), "btemp_6_9v":(11,)},
    {"nersc_sar_primary":(4,), "nersc_sar_secondary":(6,),"CT":(8,), "btemp_6_9h":(10,), "btemp_6_9v":(12,)},
    {"nersc_sar_primary":(4,), "nersc_sar_secondary":(6,),"CT":(8,), "btemp_6_9h":(10,), "btemp_6_9v":(12,)},
    {"nersc_sar_primary":(4,), "nersc_sar_secondary":(6,),"CT":(8,), "btemp_6_9h":(10,), "btemp_6_9v":(12,)},
    {"nersc_sar_primary":(4,), "nersc_sar_secondary":(6,),"CT":(8,), "btemp_6_9h":(10,), "btemp_6_9v":(12,)},
    {"nersc_sar_primary":(4,), "nersc_sar_secondary":(6,),"CT":(8,), "btemp_6_9h":(10,), "btemp_6_9v":(12,)}
                                                      ]
                )
    def test_function_data_generation(self, mock_load):
        """data must be generated from files by using 'np.loads' for this data generator"""
        Archive = mock.Mock()
        config_ = FileBasedConfigure(archive=Archive)
        config_.dims_input = (1, 1)
        config_.dims_output = (1, 1)
        config_.dims_amsr2 = (1, 1)
        config_.batch_size = 2
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [1,2,]}
        config_.set_params()
        self.test_generator = DataGeneratorFrom_npz_File(config_.partition['train'], **config_.params)
        ans1, ans2 = self.test_generator[0]
        np.testing.assert_equal(ans1, [np.array([[[[3., 5.]]], [[[4., 6.]]]]),
                                       np.array([[[[ 9., 11.]]], [[[10., 12.]]]])])
        np.testing.assert_equal(ans2, np.array([[[[7.]]], [[[8.]]]]))

class DataGeneratorFromMemoryTestCases(unittest.TestCase):
    """Tests for DataGeneratorFromMemory"""
    def test_function_data_generation(self):
        """data must be generated from config_.archive.PROP for this data generator"""
        Archive = mock.Mock()
        config_ = MemoryBasedConfigure(archive=Archive)
        config_.dims_input = (1, 1)
        config_.dims_output = (1, 1)
        config_.dims_amsr2 = (1, 1)
        config_.batch_size = 2
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [1,2]}
        config_.archive.PROP={"nersc_sar_primary":(3,4), "nersc_sar_secondary":(5,6), "CT":(7,8),
                                "btemp_6_9h":(9,10), "btemp_6_9v":(11,12), "_locs":(1,2)}
        config_.set_params()
        self.test_generator = DataGeneratorFromMemory(config_.partition['train'], **config_.params)
        ans1, ans2 = self.test_generator[0]
        np.testing.assert_equal(ans1, [np.array([[[[3., 5.]]], [[[4., 6.]]]]),
                                       np.array([[[[ 9., 11.]]], [[[10., 12.]]]])])
        np.testing.assert_equal(ans2, np.array([[[[7.]]], [[[8.]]]]))
