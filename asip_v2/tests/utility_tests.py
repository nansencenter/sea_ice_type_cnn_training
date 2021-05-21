import unittest
import unittest.mock as mock
import numpy as np
import datetime
import tempfile
from os.path import join, isdir
import sys
from utility import FileBasedConfigure, MemoryBasedConfigure,  read_input_params

class InitializationTestCases(unittest.TestCase):

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.tmp_dir = self.temp_directory.name
        self.temp_directory_2nd = tempfile.TemporaryDirectory()
        self.tmp_dir_output = self.temp_directory_2nd.name
        sys.argv = ['/workspaces/ASIP-v2-builder/asip_v2/apply_or_train_model.py',
                    self.tmp_dir, '-o', self.tmp_dir_output,
                    '-n', 'nersc_', '-w', '700', '-s', '700',
                    '--rm_swath', '0', '-d', '0', '-i', '-p', '1.',
                    '-bs', '4', '-r', '50']

    def tearDown(self):
        self.temp_directory.cleanup()
        self.temp_directory_2nd.cleanup()

    def test_initialization_archive(self):
        """archive object must be instantiated with property attributes"""
        archive_ = read_input_params()
        self.assertEqual(archive_.AMSR_LABELS,
           ['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h', 'btemp_10.7v',
           'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h', 'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v',
           'btemp_89.0h', 'btemp_89.0v']
                        )
        self.assertEqual(archive_.ASPECT_RATIO , 50)
        self.assertEqual(archive_.DATAPATH , self.tmp_dir)
        self.assertEqual(archive_.DISTANCE_THRESHOLD , 0)
        self.assertEqual(archive_.NERSC , 'nersc_')
        self.assertEqual(archive_.OUTPATH , self.tmp_dir_output)
        self.assertEqual(archive_.PROP , {})
        self.assertEqual(archive_.RM_SWATH, 0)
        self.assertEqual(archive_.SAR_NAMES, ['nersc_sar_primary', 'nersc_sar_secondary'])
        self.assertEqual(archive_.STRIDE_AMS2_SIZE, 14)
        self.assertEqual(archive_.STRIDE_SAR_SIZE, 700)
        self.assertEqual(archive_.WINDOW_SIZE, (700, 700))
        self.assertEqual(archive_.WINDOW_SIZE_AMSR2, (14, 14))
        self.assertEqual(archive_.apply_instead_of_training, True)
        self.assertEqual(archive_.batch_size, 4)
        self.assertEqual(archive_.beginning_day_of_year, 0)
        self.assertEqual(archive_.ending_day_of_year, 365)
        self.assertEqual(archive_.memory_mode, False)
        self.assertEqual(archive_.precentage_of_training, 1.0)
        self.assertEqual(archive_.shuffle_for_training, False)
        self.assertEqual(archive_.shuffle_on_epoch_end, False)
        self.assertEqual(archive_.step_output, 1)
        self.assertEqual(archive_.step_sar, 1)

    def test_initialization_config(self):
        """config object must be instantiated with property attributes"""
        archive_ = read_input_params()
        config_ = MemoryBasedConfigure(archive=archive_)
        self.assertEqual(config_.ASPECT_RATIO, 50)
        self.assertEqual(config_.BEGINNING_DAY_OF_YEAR, 0)
        self.assertEqual(config_.DATAPATH, self.tmp_dir)
        self.assertEqual(config_.ENDING_DAY_OF_YEAR, 365)
        self.assertEqual(config_.OUTPATH, self.tmp_dir_output)
        self.assertEqual(config_.WINDOW_SIZE, (700, 700))
        self.assertEqual(config_.WINDOW_SIZE_AMSR2, (14, 14))
        self.assertEqual(config_.amsr2_var_names, ['btemp_6_9h', 'btemp_6_9v'])
        self.assertEqual(config_.archive, archive_)
        self.assertEqual(config_.batch_size, 4)
        self.assertEqual(config_.extension, '.nc')
        self.assertEqual(config_.input_var_names, ['nersc_sar_primary', 'nersc_sar_secondary'])
        self.assertEqual(config_.output_var_name, 'CT')
        self.assertEqual(config_.precentage_of_training, 1.0)
        self.assertEqual(config_.shuffle_for_training, False)
        self.assertEqual(config_.shuffle_on_epoch_end, False)

        config_ = FileBasedConfigure(archive=archive_)
        self.assertEqual(config_.extension, '.npz') # extension differs between two configurations

        w=0

class ConfigureTestCases(unittest.TestCase):
    """tests for Configure methods"""
    @mock.patch('archive.Archive.__init__', return_value=None)
    @mock.patch('utility.MemoryBasedConfigure.divide_id_list_into_partition')
    @mock.patch('utility.MemoryBasedConfigure.instantiate_generator_with_params_and_associated_partition')
    @mock.patch('utility.MemoryBasedConfigure.filling_id_list')
    @mock.patch('utility.MemoryBasedConfigure.calculate_dims')
    def test_function_setup_generator(self, mock_calculate_dims, mock_filling_id_list,
                             mock_instantiate_generator, divide_id_list, mock_archive):
        """ four methods of 'setup_generator' should be called once """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.setup_generator()
        mock_calculate_dims.assert_called_once()
        mock_filling_id_list.assert_called_once()
        mock_instantiate_generator.assert_called_once()
        divide_id_list.assert_called_once()

    @mock.patch('utility.MemoryBasedConfigure.set_the_folder_of_reconstructed_files')
    @mock.patch('utility.MemoryBasedConfigure.setup_generator')
    @mock.patch('utility.MemoryBasedConfigure.create_model')
    @mock.patch('utility.MemoryBasedConfigure.predict_by_model')
    @mock.patch('utility.MemoryBasedConfigure.reconstruct_the_image_and_reset_archive_PROP')
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

    return_value = ['20180410T084537_000006_nersc_-0_6.npz', '20180410T084537_000006_nersc_-1_6.npz']
    @mock.patch('utility.FileBasedConfigure.set_the_folder_of_reconstructed_files')
    @mock.patch('utility.FileBasedConfigure.setup_generator')
    @mock.patch('utility.FileBasedConfigure.create_model')
    @mock.patch('utility.FileBasedConfigure.predict_by_model')
    @mock.patch('utility.FileBasedConfigure.reconstruct_the_image_and_reset_archive_PROP')
    @mock.patch('os.listdir', return_value=return_value)
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_apply_model_for_file_based_config(self, mock_archive,  mock_listdir, mock_reconstruct,
                                        mock_predict, mock_create, mock_setup, mock_set_the_folder):
        """full address shall be in the 'list of names' and the date shall be in 'scene_date'"""
        config_ = FileBasedConfigure(archive=mock_archive)
        config_.DATAPATH = "/bar"
        config_.apply_model()
        mock_set_the_folder.assert_called_once()
        mock_setup.assert_called_once()
        mock_create.assert_called_once()
        mock_predict.assert_called_once()
        mock_reconstruct.assert_called_once()
        self.assertEqual(['/bar/20180410T084537_000006_nersc_-0_6.npz',
                          '/bar/20180410T084537_000006_nersc_-1_6.npz'], config_.list_of_names)
        self.assertEqual('20180410T084537', config_.scene_date)

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_divide_id_list_into_partition(self, mock_archive):
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.id_list = ["bar","foo"]
        config_.shuffle_for_training = False
        config_.precentage_of_training = 0.5
        config_.divide_id_list_into_partition()
        self.assertEqual(config_.partition,{'train': ['bar'], 'validation': ['foo']})

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_instantiate_generator_with_params_and_associated_partition(self, mock_archive):
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
        config_.partition = {'train': ['bar'], 'validation': ['foo']}
        config_.instantiate_generator_with_params_and_associated_partition()
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
        self.assertEqual(config_.training_generator.__class__.__name__, 'DataGeneratorFromMemory')
        self.assertEqual(config_.training_generator.list_IDs, ['bar'])
        self.assertEqual(config_.validation_generator.list_IDs, ['foo'])

    @mock.patch('utility.np.savez')
    @mock.patch('utility.MemoryBasedConfigure.instantiate_image_with_zeros_and_get_the_patch_locations_of_image')
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

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_set_the_folder_of_reconstructed_files(self, mock_archive):
        """
        1. shall return the proper name in 'reconstruct_path' attribue with "reconstructs_folder"
        at the end of the name.
        2. shall create a folder with this name (one level up in foldering for file based config)
        """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.DATAPATH = tempfile.TemporaryDirectory().name #e.g. DATAPATH='/tmp/tmp1l8y0cxj'
        config_.set_the_folder_of_reconstructed_files()
        self.assertEqual(config_.reconstruct_path, join(config_.DATAPATH, "reconstructs_folder"))
        self.assertTrue(isdir(join(config_.DATAPATH, "reconstructs_folder")))

        config_ = FileBasedConfigure(archive=mock_archive)
        config_.DATAPATH = tempfile.TemporaryDirectory().name #e.g. DATAPATH='/tmp/tmp2l7y0cbv'
        config_.set_the_folder_of_reconstructed_files()
        self.assertEqual(config_.reconstruct_path, '/tmp/reconstructs_folder')
        self.assertTrue(isdir('/tmp/reconstructs_folder'))

class FileBasedConfigureTestCases(unittest.TestCase):
    """ Tests for FileBasedConfigure"""
    @mock.patch('utility.np.load', return_value={'nersc_sar_primary':np.zeros([10, 20]),
                                                 'CT':np.zeros([30, 40]),
                                                 'btemp_6_9h':np.zeros([50, 60])})
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_calculate_dims(self, mock_archive, mock_np_load):
        """ shall set the correct dims """
        config_ = FileBasedConfigure(archive=mock_archive)
        config_.id_list = [""]
        config_.calculate_dims()
        self.assertEqual(config_.dims_input, (10, 20))
        self.assertEqual(config_.dims_output, (30, 40))
        self.assertEqual(config_.dims_amsr2, (50, 60))

    @mock.patch('archive.Archive.__init__', return_value=None)
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

    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_instantiate_image(self, mock_archive):
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

    @mock.patch('utility.Dataset', return_value=None)
    @mock.patch('utility.basename', return_value=None)
    @mock.patch('archive.Archive')
    def test_function_filling_id_list(self, mock_archive, mock_basename, mock_Dataset):
        """ shall set the id list properly """
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.list_of_names = []
        mock_archive.PROP = {'_locs': [(0, 0), (0, 1), (0, 2), (0, 3)]}
        config_.filling_id_list()
        config_.archive.calculate_PROP_of_archive.assert_called_once()
        self.assertEqual(config_.id_list, [(0, 0), (0, 1), (0, 2), (0, 3)])

    @mock.patch('utility.Dataset', return_value={'btemp_6.9h':np.zeros([8,9])})
    @mock.patch('archive.Archive.__init__', return_value=None)
    def test_function_instantiate_image(self, mock_archive, mock_Dataset):
        """ shall instantiate the image array with proper size and set the patch locations"""
        config_ = MemoryBasedConfigure(archive=mock_archive)
        config_.list_of_names = ['']
        config_.ASPECT_RATIO = 10
        mock_archive.PROP = {'_locs': [(0, 0), (0, 1), (0, 2), (0, 3)]}
        config_.instantiate_image_with_zeros_and_get_the_patch_locations_of_image()
        self.assertEqual(config_.img.shape, (80, 90))
        self.assertEqual(config_.patch_locs, [(0, 0), (0, 1), (0, 2), (0, 3)])
