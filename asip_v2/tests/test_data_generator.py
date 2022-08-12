import unittest
import unittest.mock as mock
import json
import numpy as np

from data_generator import (DataGenerator, HugoDataGenerator, HugoBinaryGenerator, DataGenerator_sod_f)
from utility import Archive, Configure


class DataGeneratorTestCases(unittest.TestCase):
    """ """
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

    def test_function_convert(self):
        test1 = [1, 0, 0, 0]
        test2 = [0.2, 0, 0, 0.4, 0, 0.1, 0, 0, 0.1]
        np.testing.assert_equal(self.test_generator.convert(test1), [1, 0, 0, 0])
        np.testing.assert_equal(self.test_generator.convert(test2), [0.2, 0, 0, 0.4, 0, 0.1, 0, 0, 0.1])

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
        """return the good type depending the stage of developpement"""
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
        np.testing.assert_equal(ans1, np.array([[[[1., 3.]]], [[[5., 7.]]]]))
        np.testing.assert_equal(ans2, np.array([[1., 0., 0., 0.], [0.0, 0.5, 0.4, 0.1]]))

class HugoBinaryGeneratorTestCases(unittest.TestCase):
    """Tests for Hugobinarygenerator"""

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
        self.test_generator = HugoBinaryGenerator(config_.partition['train'], **config_.params)

    def tearDown(self):
        del self.test_generator

    def test_function_convert(self):
        test1 = [0, 0, 1.0, 0, 0]
        test2 = [1, 0, 0, 0, 0]
        test3 = [0.0, 0.5, 0.4, 0.1, 0]
        test4 = [0.1, 0, 0.9, 0, 0]
        test5 = [0.2, 0, 0.2, 0.6, 0]
        test6 = [0.2, 0, 0.4, 0.4, 0]
        test7 = [0.6, 0.3 , 0.1, 0, 0]
        test8 = [0.5, 0.2 , 0.1 , 0.2, 0]
        test9 = [0.4, 0.2 , 0.2, 0.2, 0]
        test10 = [0.6, 0.2 , 0.2 , 0, 0]
        test11 = [0.5, 0.2 , 0.2 , 0, 0.1]
        test12 = [0.4, 0.2 , 0.1 , 0, 0.3]
        np.testing.assert_equal(self.test_generator.convert(test1), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.convert(test2), [1, 0, 0, 0])
        np.testing.assert_equal(self.test_generator.convert(test3), [0, 1, 0, 0])
        np.testing.assert_equal(self.test_generator.convert(test4), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.convert(test5), [0, 0, 0, 1])
        np.testing.assert_equal(self.test_generator.convert(test6), [0, 0, 0, 1])
        np.testing.assert_equal(self.test_generator.convert(test7), [0, 1, 0, 0])
        np.testing.assert_equal(self.test_generator.convert(test8), [0, 0, 0, 1])
        np.testing.assert_equal(self.test_generator.convert(test9), [0, 0, 0, 1])
        np.testing.assert_equal(self.test_generator.convert(test10), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.convert(test11), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.convert(test12), [0, 0, 0, 0])

    def test_function_one_hot_continous(self):
        """return the good type depending the stage of developpement"""
        test1_hugo = np.array([91, 60, 93,  7, 40, 91,  6, -9, -9, -9])
        test2_hugo =  np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])
        test3_hugo =  np.array([91, 50, 85, 4, 40, 91, 6, 10, 97, 7])
        test4_hugo = np.array([92, -9, 91,  8, -9, -9, -9, -9, -9, -9])
        test5_hugo = np.array([50, 20, 95, 3, 20, 93, 2, 10, 91, 2])
        test6_hugo = np.array([50, 20, 81, 3, 20, 82, 2, 10, 91, 2])
        test7_hugo = np.array([50, 20, 81, 3, 20, 92, 2, 10, 91, 2])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test1_hugo), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test2_hugo), [1, 0, 0, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test3_hugo), [0, 1, 0, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test4_hugo), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test5_hugo), [0, 0, 1, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test6_hugo), [0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator.one_hot_continous(test7_hugo), [0, 0, 1, 0])

class DataGenerator_sod_fTest_case(unittest.TestCase):
    """Tests for Datagenerator_sod_f"""

    def setUp(self):
        Archive = mock.Mock()
        config_ = Configure(archive=Archive)
        config_.dims_input = (50, 50, 4)
        config_.dims_amsr2 = (14, 14)
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
        self.assertEqual(self.test_generator3.X.shape, (2, 50, 50, 4))
        self.assertEqual(self.test_generator3.y.shape, (2, 17))
        self.assertEqual(self.test_generator3.z.shape, (2, 14, 14, 2))

    def test_function_one_hot_continous(self):
        """return the good type depending the stage of developpement"""
        self.test_generator3.list_combi = list_combi =["0_0", "82_2_3_4_5_83_3_4", "83_5", "83_6_87_3_4_5", "87_6", "91_2_3_4", "91_5", "91_6", "91_7", "93_2",
             "93_3_4_5", "93_6", "93_7","95_3_4", "95_5", "95_6_7", "96_6_97_7"]
        test1_sod_f = np.array([91, 60, 95, 3, 40, 91, 5, -9, -9, -9])
        test2_sod_f =  np.array([2, -9, -9,  -9, -9, -9,  -9, -9, -9, -9])
        test3_sod_f =  np.array([91, 50, 83, 5, 40, 87, 6, 10, 95, 6])
        test4_sod_f = np.array([92, -9, 91,  6, -9, -9, -9, -9, -9, -9])
        test5_sod_f = np.array([80, 30, 97,  7, 40, 96,  6, 10, 93,  6])
        test6_sod_f = np.array([10, -9, 83,  5, -9, -9, -9, -9, -9, -9])
        test7_sod_f = np.array([90, 70, 91,  6, 20, 87,  5, -9, -9, -9])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test1_sod_f), [0, 0, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test2_sod_f), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test3_sod_f), [0, 0, 0.5, 0, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test4_sod_f), [0.1, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test5_sod_f), [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.7])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test6_sod_f), [0.9, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(self.test_generator3.one_hot_continous(test7_sod_f), [0.1, 0, 0, 0.2, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])


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
        self.test_generator2.dims_input = (1, 1, 4)
        # print(self.test_generator2[0])
        ans0, ans3 = self.test_generator2[0]
        ans1 = ans0[0]
        ans2= ans0[1]
        np.testing.assert_equal(ans1, np.array(([[[[1., 2., 3., 4.]]], [[[1., 2., 3., 4.]]]])))
        # np.testing.assert_equal(ans2, np.array([[[[-2.61447351, -5.5074949 ], [-2.59670627, -5.47985681]], [[-2.57893904, -5.45221873], [-2.5611718, -5.42458065]]], [[[-2.61447351, -5.5074949 ], [-2.59670627, -5.47985681]], [[-2.57893904, -5.45221873], [-2.5611718, -5.42458065]]]]))
        np.testing.assert_equal(ans3, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
