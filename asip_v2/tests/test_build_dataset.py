import sys
import unittest
import unittest.mock as mock

from build_dataset import read_input_params_for_building


class BuildDatasetTestCases(unittest.TestCase):
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_read_input_params_for_applying(self, mock_archive):
        """input from sys.argv should be read correctly"""
        sys.argv = ['', "dir_1", '-o','dir_2', '-n', 'nersc_', '-w', '700', '-s', '700', '-r', '50']
        read_input_params_for_building()
        mock_archive.assert_called_with(
            amsr_labels=['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h',
                         'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h', 'btemp_23.8v',
                         'btemp_36.5h', 'btemp_36.5v', 'btemp_89.0h', 'btemp_89.0v'],
            apply_instead_of_training=False, aspect_ratio=50, datapath='dir_1', distance_threshold=0,
            nersc='nersc_', outpath='dir_2', rm_swath=0, step_output=1, step_sar=1,
            stride_ams2_size=14, stride_sar_size=700, window_size=(700, 700),
            window_size_amsr2=(14, 14), sar_names=['nersc_sar_primary', 'nersc_sar_secondary']
                                       )
