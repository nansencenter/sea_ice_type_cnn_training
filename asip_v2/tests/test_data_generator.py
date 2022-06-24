import unittest
import unittest.mock as mock
import json
import numpy as np

from data_generator import (DataGenerator, HugoDataGenerator, DataGenerator_sod_f)
from utility import Archive, Configure


class DataGeneratorTestCases(unittest.TestCase):
    """Tests for Datagenerator """
    def setUp(self):
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.dims_amsr2 = (16, 16, 4)
        config_.input_var_names = ["input_var_name1", "input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1", "amsr2_var_names2"]
        config_.batch_size = 2
        config_.idir_json = './tests'
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [56, 87, 34, 12, 90, 75, 88]}# a list with length of seven
        config_.set_params()
        self.test_generator = DataGenerator(config_.partition['train'], **config_.params)

    def tearDown(self):
        del self.test_generator

    def test_len(self):
        self.assertEqual(len(self.test_generator), 3) # 7//2=3

    def test_function_on_epoch_end(self):
        self.test_generator.on_epoch_end()
        # regardless of values inside the 'train' list, the indexes must be the array containing
        # the numbers up to len of it
        np.testing.assert_equal(self.test_generator.indexes, np.array([0, 1, 2, 3, 4, 5, 6]))

        
class HugoDataGeneratorTestCases(unittest.TestCase):
    """Tests for HugoDatagenerator"""

    def setUp(self):
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.dims_input = (50, 50, 2)
        config_.dims_amsr2 = (16, 16, 4)
        config_.input_var_names = ["input_var_name1", "input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1", "amsr2_var_names2"]
        config_.batch_size = 2
        config_.idir_json = './tests'
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [56, 87, 34, 12, 90, 75, 88]}# a list with length of seven
        config_.set_params()
        self.test_generator = HugoDataGenerator(config_.partition['train'], **config_.params)

    def tearDown(self):
        del self.test_generator

    def test_function_x_y_z_initialization(self):
        self.test_generator.x_y_z_initialization()
        self.assertEqual(self.test_generator.X.shape, (2, 50, 50, 2))
        self.assertEqual(self.test_generator.y.shape, (2, 4))

    def test_function_ice_type(self):
        """return the good type depending the stage of developpement"""
        test1_stage = 2
        test2_stage = 83
        test3_stage = 92
        test4_stage = 97
        np.testing.assert_equal(self.test_generator.ice_type(test1_stage), 0)
        np.testing.assert_equal(self.test_generator.ice_type(test2_stage), 1)
        np.testing.assert_equal(self.test_generator.ice_type(test3_stage), 2)
        np.testing.assert_equal(self.test_generator.ice_type(test4_stage), 3)

    def test_function_one_hot_continous(self):
        """return the good vector according to combinations"""
        test1_hugo = np.array([91, 60, 93,  7, 40, 91,  6, -9, -9, -9])
        test2_hugo =  np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])
        test3_hugo =  np.array([91, 50, 85, 4, 40, 91, 6, 10, 97, 7])
        test4_hugo = np.array([92, -9, 91,  8, -9, -9, -9, -9, -9, -9])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test1_hugo), [0, 0, 1.0, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test2_hugo), [1, 0, 0, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test3_hugo), [0.0, 0.5, 0.4, 0.1])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test4_hugo), [0.1, 0, 0.9, 0])

    @mock.patch('data_generator.np.load', side_effect=[
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 85, 4, 40, 91, 6, 10, 97, 7])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 85, 4, 40, 91, 6, 10, 97, 7])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 85, 4, 40, 91, 6, 10, 97, 7])}
     ])
    def test_function_data_generation(self, mock_load):
        """data must be generated from files by using 'np.loads' for this data generator"""
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ['btemp_6_9h','btemp_6_9v']
        config_.dims_amsr2 = (1, 1)
        config_.batch_size = 2
        config_.idir_json = './tests'
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [1,2,]}
        config_.set_params()
        self.test_generator2 = HugoDataGenerator(config_.partition['train'], **config_.params)
        self.test_generator2.dims_input = (1, 1, 2)
        ans1, ans2 = self.test_generator2[0]
        np.testing.assert_equal(ans1, [np.array([[[[1., 3.]]], [[[5., 7.]]]])])
        np.testing.assert_equal(ans2, np.array([[1., 0., 0., 0.], [0.0, 0.5, 0.4, 0.1]]))


class DataGenerator_sod_fTest_case(unittest.TestCase):
    """Tests for Datagenerator_sod_f"""

    def setUp(self):
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.dims_input = (50, 50, 2, 2)
        config_.dims_amsr2 = (16, 16)
        config_.input_var_names = ["input_var_name1", "input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1", "amsr2_var_names2"]
        config_.batch_size = 2
        config_.idir_json = './tests'
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [56, 87, 34, 12, 90, 75, 88]}# a list with length of seven
        config_.set_params()
        self.test_generator3 = DataGenerator_sod_f(config_.partition['train'], **config_.params)

    def tearDown(self):
        del self.test_generator3

    def test_function_x_y_z_initialization(self):
        self.test_generator3.x_y_z_initialization()
        y_len = len(self.test_generator3.list_combi)
        self.assertEqual(self.test_generator3.X.shape, (2, 50, 50, 2, 2))
        self.assertEqual(self.test_generator3.y.shape, (2, y_len))
        self.assertEqual(self.test_generator3.z.shape, (2, 16, 16, 2))

    def test_function_one_hot_continous(self):
        """eturn the good vector according to combinations"""
        self.test_generator3.list_combi = ["0_0", "83_5", "93_6", "87_6", "95_4", "95_6", "91_5", "95_3", "95_5", "97_7", "96_6", "91_6", "87_5"]
        test1_sod_f = np.array([91, 60, 95, 3, 40, 91, 5, -9, -9, -9])
        test2_sod_f =  np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])
        test3_sod_f =  np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6])
        test4_sod_f = np.array([92, -9, 91,  6, -9, -9, -9, -9, -9, -9])
        test5_sod_f = np.array([80, 30, 97,  7, 40, 96,  6, 10, 93,  6])
        test6_sod_f = np.array([10, -9, 83,  5, -9, -9, -9, -9, -9, -9])
        test7_sod_f = np.array([90, 70, 91,  6, 20, 87,  5, -9, -9, -9])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test1_sod_f), [0.0, 0, 0, 0, 0, 0, 0.4, 0.6, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test2_sod_f), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test3_sod_f), [0.0, 0.5, 0, 0.4, 0, 0.1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test4_sod_f), [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test5_sod_f), [0.2, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.3, 0.4, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test6_sod_f), [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test7_sod_f), [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.2])

    @mock.patch('data_generator.np.load', side_effect=[
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9]), "btemp_6_9h":np.array([[1,2],[3,4]]), "btemp_6_9v":np.array([[5,6],[7,8]])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9]), "btemp_6_9h":np.array([[1,2],[3,4]]), "btemp_6_9v":np.array([[5,6],[7,8]])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9]), "btemp_6_9h":np.array([[1,2],[3,4]]), "btemp_6_9v":np.array([[5,6],[7,8]])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9]), "btemp_6_9h":np.array([[1,2],[3,4]]), "btemp_6_9v":np.array([[5,6],[7,8]])},
    {"nersc_sar_primary":np.array([[[1,2]]]), "nersc_sar_secondary":np.array([[[3,4]]]),"CT":np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9]), "btemp_6_9h":np.array([[1,2],[3,4]]), "btemp_6_9v":np.array([[5,6],[7,8]])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6]), "btemp_6_9h":np.array([[9,10],[11,12]]), "btemp_6_9v":np.array([[13,14],[15,16]])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6]), "btemp_6_9h":np.array([[9,10],[11,12]]), "btemp_6_9v":np.array([[13,14],[15,16]])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6]), "btemp_6_9h":np.array([[9,10],[11,12]]), "btemp_6_9v":np.array([[13,14],[15,16]])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6]), "btemp_6_9h":np.array([[9,10],[11,12]]), "btemp_6_9v":np.array([[13,14],[15,16]])},
    {"nersc_sar_primary":np.array([[[5,6]]]), "nersc_sar_secondary":np.array([[[7,8]]]),"CT":np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6]), "btemp_6_9h":np.array([[9,10],[11,12]]), "btemp_6_9v":np.array([[13,14],[15,16]])}
     ])
    def test_function_data_generation(self, mock_load):
        """data must be generated from files by using 'np.loads' for this data generator"""
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.input_var_names = ['nersc_sar_primary', 'nersc_sar_secondary']
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ['btemp_6_9h','btemp_6_9v']
        config_.dims_amsr2 = (2, 2)
        config_.batch_size = 2
        config_.idir_json = './tests'
        config_.shuffle_on_epoch_end = False
        config_.partition = {'train': [1,2]}
        config_.set_params()
        self.test_generator2 = DataGenerator_sod_f(config_.partition['train'], **config_.params)
        self.test_generator2.dims_input = (1, 1, 2, 2)
        # print(self.test_generator2[0])
        ans0, ans3 = self.test_generator2[0]
        ans1 = ans0[0]
        ans2= ans0[1]
        np.testing.assert_equal(ans1, np.array([[[[[1, 2], [3, 4]]]], [[[[5, 6],[7, 8]]]]]))
        np.testing.assert_equal(ans2, np.array([[[[1, 5], [2, 6]], [[3, 7], [4, 8]]], [[[9, 13], [10, 14]], [[11, 15], [12, 16]]]]))
        np.testing.assert_equal(ans3, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0.5, 0, 0.4, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
