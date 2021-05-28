import unittest
import unittest.mock as mock
import numpy as np
from os.path import join, isdir
import sys
import tempfile
from apply_model import MemoryBasedConfigure, read_input_params_for_applying

class MemoryBasedConfigureTestCases(unittest.TestCase):
    """ Tests for MemoryBasedConfigure"""
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_calculate_dims(self, mock_archive):
        """ shall set the correct dims """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.WINDOW_SIZE = (2, 3)
        config_.WINDOW_SIZE_AMSR2 = (4, 5)
        config_.calculate_dims()
        self.assertEqual(config_.dims_output, (2, 3))
        self.assertEqual(config_.dims_input, (2, 3))
        self.assertEqual(config_.dims_amsr2, (4, 5))

    @mock.patch('apply_model.Dataset', return_value=None)
    @mock.patch('apply_model.basename', return_value=None)
    @mock.patch('archive.Archive')
    def test_function_filling_id_list(self, mock_archive, mock_basename, mock_Dataset):
        """ shall set the id list properly """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.list_of_names = []
        mock_archive.PROP = {'_locs': [(0, 0), (0, 1), (0, 2), (0, 3)]}
        config_.filling_id_list()
        config_.archive.process_dataset.assert_called_once()
        self.assertEqual(config_.id_list, [(0, 0), (0, 1), (0, 2), (0, 3)])

    @mock.patch('apply_model.Dataset', return_value={'btemp_6.9h':np.zeros([8,9])})
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_instantiate_image(self, mock_archive, mock_Dataset):
        """ shall instantiate the image array with proper size and set the patch locations"""
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.list_of_names = ['']
        config_.ASPECT_RATIO = 10
        mock_archive.PROP = {'_locs': [(0, 0), (0, 1), (0, 2), (0, 3)]}
        config_.instantiate_image_with_zeros_and_get_the_patch_locations_of_image()
        # must be the multiplication of return_value of 'btemp_6.9h' and aspect ratio
        self.assertEqual(config_.img.shape, (80, 90))
        self.assertEqual(config_.patch_locs, [(0, 0), (0, 1), (0, 2), (0, 3)])

    @mock.patch('apply_model.np.savez')
    @mock.patch('apply_model.MemoryBasedConfigure.instantiate_image_with_zeros_and_get_the_patch_locations_of_image')
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_reconstruct_the_image_and_reset_archive_PROP(self, mock_archive,
                                                                   mock_instantiate,
                                                                    mock_savez
                                                                    ):
        """" assembled version of y_pred must be written to npz file. """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.reconstruct_path = "/bar"
        config_.scene_date = '20180410T084537'
        config_.WINDOW_SIZE = (1, 1)
        config_.patch_locs = [(0, 0), (0, 1), (1,0), (1, 1)]
        config_.img = np.zeros([2, 2])
        config_.y_pred = np.array([[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]])
        config_.reconstruct_the_image_and_reset_archive_PROP()

        mock_savez.assert_called_once()
        # must have the correct path and suffix (as well as extension) of the file
        self.assertEqual(mock_savez.call_args[0][0], '/bar/20180410T084537_reconstruct.npz')
        # must be called with the assembled version of above array.
        np.testing.assert_array_equal(mock_savez.call_args[0][1], np.array([[1., 2.], [3., 4.]]))
        # archive PROP must be reset at the end
        self.assertEqual(config_.archive.PROP, {})


    @mock.patch('apply_model.MemoryBasedConfigure.set_the_folder_of_reconstructed_files')
    @mock.patch('apply_model.MemoryBasedConfigure.setup_generator')
    @mock.patch('apply_model.MemoryBasedConfigure.create_model')
    @mock.patch('apply_model.MemoryBasedConfigure.predict_by_model')
    @mock.patch('apply_model.MemoryBasedConfigure.reconstruct_the_image_and_reset_archive_PROP')
    @mock.patch('os.listdir', return_value=['20190411T084522_.nc'])
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_apply_model_for_memory_based_config(self, mock_archive,  mock_listdir, mock_reconstruct,
                                        mock_predict, mock_create, mock_setup, mock_set_the_folder):
        """full address shall be in the 'list of names' and the date shall be in 'scene_date'"""
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.DATAPATH = "/bar"
        config_.apply_model()
        mock_set_the_folder.assert_called_once()
        mock_setup.assert_called_once()
        mock_create.assert_called_once()
        mock_predict.assert_called_once()
        mock_reconstruct.assert_called_once()
        self.assertEqual(['/bar/20190411T084522_.nc'], config_.list_of_names)
        self.assertEqual('20190411T084522', config_.scene_date)

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_set_the_folder_of_reconstructed_files(self, mock_archive):
        """
        1. shall return the proper name in 'reconstruct_path' attribue with "reconstructs_folder"
        at the end of the name.
        2. shall create a folder with this name (one level up in foldering for file based config)
        """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        temp_path = tempfile.TemporaryDirectory()
        config_.DATAPATH = temp_path.name #e.g. DATAPATH='/tmp/tmp1l8y0cxj'
        config_.set_the_folder_of_reconstructed_files()
        self.assertEqual(config_.reconstruct_path, join(config_.DATAPATH, "reconstructs_folder"))
        self.assertTrue(isdir(join(config_.DATAPATH, "reconstructs_folder")))
        temp_path.cleanup()

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_set_params(self, mock_archive):
        """config_.params should be set correctly"""
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.dims_input = (700, 700)
        config_.dims_output = (700, 700)
        config_.dims_amsr2 = (14, 14)
        config_.input_var_names = ["input_var_name1","input_var_name2"]
        config_.output_var_name = 'CT'
        config_.amsr2_var_names = ["amsr2_var_names1","amsr2_var_names2"]
        config_.batch_size = 10
        config_.shuffle_on_epoch_end = True
        config_.archive.PROP = {}
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
                                          'prop': {}
                                         }
                        )

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_read_input_params_for_applying(self, mock_archive):
        """input from sys.argv should be read correctly"""
        sys.argv = ['', "dir_name", '-n', 'nersc_', '-w', '700', '-s', '700', '-r', '50', '-bs','4']
        read_input_params_for_applying()
        mock_archive.assert_called_with(
            amsr_labels=['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h',
                         'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h', 'btemp_23.8v',
                         'btemp_36.5h', 'btemp_36.5v', 'btemp_89.0h', 'btemp_89.0v'],
            apply_instead_of_training=True, aspect_ratio=50, batch_size=4, datapath='dir_name',
            distance_threshold=0, nersc='nersc_', percentage_of_training=1.0, rm_swath=0,
            sar_names=['nersc_sar_primary', 'nersc_sar_secondary'], shuffle_for_training=False,
            shuffle_on_epoch_end=False, step_output=1, step_sar=1, stride_ams2_size=14,
            stride_sar_size=700, window_size=(700, 700), window_size_amsr2=(14, 14)
        )
