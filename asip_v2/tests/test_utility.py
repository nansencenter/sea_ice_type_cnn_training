import argparse
import datetime
import sys
import tempfile
import unittest
import unittest.mock as mock
from os.path import isdir, join

import numpy as np
from utility import (Configure, between_zero_and_one_float_type, common_parser,
                     postprocess_the_args, type_for_nersc_noise)


class UtilityFunctionsTestCases(unittest.TestCase):
    """Tests for functions inside utilities"""

    def test_common_parser_and_postprocess_the_args(self):
        """ common parser shall correctly set the 'dict_for_archive_init' """
        sys.argv = ['', "dir_name", '-n', 'nersc_', '-w', '700', '-s', '700', '-r', '50']
        parser = common_parser()
        arg = parser.parse_args()
        dict_for_archive_init = postprocess_the_args(arg)
        self.assertEqual(dict_for_archive_init,
            {'sar_names': ['nersc_sar_primary', 'nersc_sar_secondary'],
            'nersc': 'nersc_',
            'datapath': "dir_name",
            'window_size': (700, 700),
            'window_size_amsr2': (14, 14),
            'stride_sar_size': 700,
            'stride_ams2_size': 14,
            'step_sar': 1,
            'step_output': 1,
            'aspect_ratio': 50,
            'rm_swath': 0,
            'amsr_labels': ['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h',
                           'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h',
                           'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v', 'btemp_89.0h', 'btemp_89.0v'],
            'distance_threshold': 0}
        )

    def test_postprocess_the_args_for_validation_window_size(self):
        """Window size must be dividable to value of aspect_ratio"""
        parser = common_parser()
        sys.argv = ['', "dir_name", '-n', 'nersc_', '-w', '701', '-s', '700', '-r', '50']
        arg = parser.parse_args()
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            postprocess_the_args(arg)
        the_exception = cm.exception
        self.assertEqual(*the_exception.args,
                                      'Window size must be dividable to value of aspect_ratio = 50')

    def test_postprocess_the_args_for_validation_stride(self):
        """Stride must be dividable to value of aspect_ratio"""
        parser = common_parser()
        sys.argv = ['', "dir_name", '-n', 'nersc_', '-w', '700', '-s', '701', '-r', '50']
        arg = parser.parse_args()
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            postprocess_the_args(arg)
        the_exception = cm.exception
        self.assertEqual(*the_exception.args,
                                           'Stride must be dividable to value of aspect_ratio = 50')

    def test_function_between_zero_and_one_float_type(self):
        """
        Anything except a float within the predefined bounds as input of funcion shall raise errors
        """
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            between_zero_and_one_float_type("")
        the_exception = cm.exception
        self.assertEqual(*the_exception.args, 'Must be a floating point number')
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            between_zero_and_one_float_type(1.5)
        the_exception = cm.exception
        self.assertEqual(*the_exception.args, 'Argument must be =< 1.0 and > 0.0')

    def test_function_type_for_nersc_noise(self):
        """
        Anything except "" or "nersc_" as input of funcion shall raise errors
        """
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            type_for_nersc_noise("foo")
        the_exception = cm.exception
        self.assertEqual(*the_exception.args, "'--noise_method' MUST be '' or 'nersc_'.")
        self.assertEqual(type_for_nersc_noise("nersc_"), "nersc_")
        self.assertEqual(type_for_nersc_noise(""), "")


class ConfigureTestCases(unittest.TestCase):
    """tests for Configure methods in utilities"""

    @mock.patch('utility.Archive.__init__', return_value=None)
    @mock.patch('utility.Configure.set_params')
    @mock.patch('utility.Configure.divide_id_list_into_partition')
    @mock.patch('utility.Configure.instantiate_generators_with_associated_partition')
    @mock.patch('utility.Configure.filling_id_list')
    @mock.patch('utility.Configure.calculate_dims')
    def test_function_setup_generator(self, mock_calculate_dims, mock_filling_id_list,
                             mock_instantiate_generator, divide_id_list, mock_set_para, mock_archive):
        """ five methods of 'setup_generator' should be called once """
        config_ = Configure(archive=mock_archive)
        config_.setup_generator()
        mock_calculate_dims.assert_called_once()
        mock_filling_id_list.assert_called_once()
        mock_instantiate_generator.assert_called_once()
        divide_id_list.assert_called_once()
        mock_set_para.assert_called_once()

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_divide_id_list_into_partition(self, mock_archive):
        """ Test the divition action of id_list into 'train' and 'validation' parts """
        config_ = Configure(archive=mock_archive)
        config_.id_list = ["bar", "foo"]
        config_.shuffle_for_training = False
        config_.precentage_of_training = 0.5
        config_.divide_id_list_into_partition()
        self.assertEqual(config_.partition, {'train': ['bar'], 'validation': ['foo']})

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_set_params(self, mock_archive):
        """self.params must be set correctly for the class"""
        config_ = Configure(archive=mock_archive)
        config_.dims_input = (700, 700)
        config_.dims_output = (700, 700)
        config_.dims_amsr2 = (14, 14)
        config_.input_var_names = ["input_var_name1","input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1","amsr2_var_names2"]
        config_.batch_size = 10
        config_.shuffle_on_epoch_end = True
        config_.set_params()
        self.assertEqual(config_.params, {
                                          'dims_input': (700, 700, 2),
                                          'dims_output': (700, 700, 1),
                                          'dims_amsr2': (14, 14, 2),
                                          'output_var_name': 'CT',
                                          'input_var_names': ['input_var_name1', 'input_var_name2'],
                                          'amsr2_var_names': ['amsr2_var_names1', 'amsr2_var_names2'],
                                          'batch_size': 10,
                                          'shuffle_on_epoch_end': True,
                                         }
                        )

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_instantiate_generator(self, mock_archive):
        """generators must be instantiated with associated partition"""
        config_ = Configure(archive=mock_archive)
        config_.partition = {'train': ['bar'], 'validation': ['foo']}
        config_.params = {'fake_param':''}
        config_.DataGenerator_ = mock.Mock()
        config_.instantiate_generators_with_associated_partition()
        #for training generator
        self.assertEqual(config_.DataGenerator_.call_args_list[0],mock.call(['bar'], fake_param=''))
        #for validation generator
        self.assertEqual(config_.DataGenerator_.call_args_list[1],mock.call(['foo'], fake_param=''))
