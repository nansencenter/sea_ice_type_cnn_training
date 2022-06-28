import sys
import unittest
import unittest.mock as mock
from unittest.mock import PropertyMock

import netCDF4 as nc
from build_dataset import Archive, main, read_input_params_for_building


class BuildDatasetTestCases(unittest.TestCase):
    """Test of BuildDataset """

    @mock.patch('build_dataset.Archive.__init__', return_value=None)
    def test_function_read_input_params_for_applying(self, mock_archive):
        """input from sys.argv should be read correctly"""
        sys.argv = ['', "dir_name", "output", '-n', 'nersc_sar', '-w', '700', '-s', '700', '-r', '50', '-d', '100']
        read_input_params_for_building()
        mock_archive.assert_called_with(input_dir='dir_name',
                                        output_dir='output',
                                        names_sar=['nersc_sar_primary', 'nersc_sar_secondary'],
                                        names_amsr2=['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h', 'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h', 'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v', 'btemp_89.0h', 'btemp_89.0v'],
                                        window_sar=700,
                                        window_amsr2=16,
                                        stride_sar=700,
                                        stride_amsr2=16,
                                        resample_step_amsr2=43,
                                        resize_step_sar=50,
                                        rm_swath=0,
                                        distance_threshold=100,
                                        encoding='continous'
                                       )


#     @mock.patch('build_dataset.Archive.update_processed_files')
#     @mock.patch('build_dataset.Archive.write_batches')
#     @mock.patch('build_dataset.Archive.process_dataset')
#     @mock.patch("build_dataset.nc.Dataset", return_value="foo")
#     def test_function_main(self,mock_dataset,mock_process,mock_write,mock_update):
#         """main should call the right functions"""
#         sys.argv = ['', "dir_name", "output", '-n', 'nersc_sar', '-w', '700', '-s', '700', '-r', '50', '-d', '100']
#         with mock.patch.object(Archive,"get_unprocessed_files") as mock_get_unprocessed:
#             main()
#         mock_get_unprocessed.assert_called_once()
#         mock_process.assert_called_once_with('foo', 'fake_file')
#         mock_write.assert_called_once()
#         mock_update.assert_called_once()
