import sys
import unittest
import unittest.mock as mock
from unittest.mock import PropertyMock

import netCDF4 as nc
from build_dataset import Archive, main, read_input_params_for_building


class BuildDatasetTestCases(unittest.TestCase):
    @mock.patch('build_dataset.Archive.__init__', return_value=None)
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

    @mock.patch('build_dataset.Archive.update_processed_files')
    @mock.patch('build_dataset.Archive.write_batches')
    @mock.patch('build_dataset.Archive.process_dataset')
    @mock.patch("build_dataset.nc.Dataset", return_value="foo")
    def test_function_main(self,mock_dataset,mock_process,mock_write,mock_update):
        """main should call the right functions"""
        sys.argv = ['', "dir_1", '-o','dir_2', '-n', 'nersc_', '-w', '700', '-s', '700', '-r', '50']
        with mock.patch.object(Archive,"get_unprocessed_files") as mock_get_unprocessed:
            mock_get_unprocessed.side_effect = __class__.side_effect_function(Archive)
            main()
        mock_get_unprocessed.assert_called_once()
        mock_process.assert_called_once_with('foo', 'fake_file')
        mock_write.assert_called_once()
        mock_update.assert_called_once()

    def side_effect_function(x):
        x.files=['fake_file']
