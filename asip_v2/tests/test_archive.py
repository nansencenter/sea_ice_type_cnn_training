import tempfile
import unittest
import unittest.mock as mock

import numpy as np
from archive import Amsr2Batches, Archive, Batches, OutputBatches, SarBatches

from skimage.util.shape import view_as_windows


class BatchesTestCases(unittest.TestCase):
    """tests for batches class"""


    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_conventer(self, mock_archive):
        """name_conventer should convert the name"""
        test_batch = Batches()
        self.assertEqual(test_batch.name_conventer("btemp_89.0h"), "btemp_89.0h")


    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_for_getdata(self, mock_archive):
        """name_for_getdata should convert the name"""
        test_batch = Batches()
        self.assertEqual(test_batch.name_for_getdata("btemp_89.0h"), "btemp_89.0h")

    @mock.patch("archive.view_as_windows")
    def test_function_view_as_windows(self, mock_view_as_window):
        """view_as_windows shall be called correctly"""
        test_batch = Batches()
        array = np.array([[1, 2], [3, 4]])
        test_batch.window=2
        test_batch.stride = 4
        test_batch.view_as_windows(array)
        mock_view_as_window.assert_called_with(array, (2, 2), 4)

    def test_function_convert(self):
        """first argument must be delivered with 'convert' function"""
        test_batch = Batches()
        self.assertEqual(test_batch.convert(1), 1)


    def test_function_resample(self):
        """first argument must be delivered with 'resample' function"""
        test_batch = Batches()
        self.assertEqual(test_batch.resample(1,1), 1)


    def test_function_resize(self):
        """
        test resizing. Resizing with step = 1 should do nothing. But for step > 1,it should resize
        """
        test_batch = Batches()
        test_batch.step = 1
        array = np.array([[1, 2], [3, 4]])
        # for dividable case
        np.testing.assert_equal(test_batch.resize(array), np.array([[1, 2], [3, 4]]))
        test_batch.step = 2
        array = np.arange(25).reshape(5, 5)
        #array=np.array([[ 0,  1,  2,  3,  4],
        #                [ 5,  6,  7,  8,  9],
        #                [10, 11, 12, 13, 14],
        #                [15, 16, 17, 18, 19],
        #                [20, 21, 22, 23, 24]])
        ## for non-dividable case
        np.testing.assert_equal(test_batch.resize(array), np.array([[0, 2], [10, 12]]))




class SarBatchesTestCases(unittest.TestCase):
    """tests for SarBatches class"""

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_resize(self, mock_archive):
        """test resizing.uniform_filter is used before the resize function in the parent class.
        Just like 'test_function_resize' of 'BatchesTestCases', resizing should not affect anything
        when step =1. """
        test_batch = SarBatches(archive_=mock_archive)
        test_batch.step = 1
        array = np.array([[1, 2], [3, 4]])
        # for dividable case
        np.testing.assert_equal(test_batch.resize(array), np.array([[1, 2], [3, 4]]))
        test_batch.step = 2
        array = np.arange(25).reshape(5, 5)
        #array=np.array([[ 0,  1,  2,  3,  4],
        #                [ 5,  6,  7,  8,  9],
        #                [10, 11, 12, 13, 14],
        #                [15, 16, 17, 18, 19],
        #                [20, 21, 22, 23, 24]])
        # for non-dividable case
        np.testing.assert_equal(test_batch.resize(array), np.array([[2, 4], [12, 14]]))



class OutputBatchesTestCases(unittest.TestCase):
    """tests for OutputBatches class"""
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_conventer(self, mock_archive):
        """name_conventer should return the correct names"""
        test_batch = OutputBatches(archive_=mock_archive)
        self.assertEqual(test_batch.name_conventer(""), "ice_type")


    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_for_getdata(self, mock_archive):
        """name_for_getdata should always return a constant value"""
        test_batch = OutputBatches(archive_=mock_archive)
        self.assertEqual(test_batch.name_for_getdata(""), "polygon_icechart")


    # @mock.patch("archive.OutputBatches.encode_icechart")
    # @mock.patch('utility.Archive.__init__', return_value=None)
    # def test_function_convert(self, mock_archive, mock_encode):
    #     """encode_icechart function must be called with correct arguments"""
    #     test_batch = OutputBatches(archive_=mock_archive)
    #     test_batch.convert('1', "2")
    #     mock_encode.assert_called_once_with('1', "2")





class Amsr2BatchesTestCases(unittest.TestCase):
    """tests for Amsr2Batches class"""

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_conventer(self, mock_archive):
        """name_conventer should convert the name"""
        test_batch = Amsr2Batches(archive_=mock_archive)
        self.assertEqual(test_batch.name_conventer("btemp_89.0h"), "btemp_89_0h")

    @mock.patch("archive.Batches.resize")
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_resize(self, mock_archive, mock_resize):
        """resize of the parent class must be called"""
        test_batch = Amsr2Batches(archive_=mock_archive)
        self.assertEqual(test_batch.resize(1), 1)


class ArchiveTestCases(unittest.TestCase):
    """tests for Archive class"""

    def test_function_get_unprocessed_files(self):
        """ Without having a file named 'processed_files.json' in the output folder,
        'files' attribute must be filled with what comes out of litsdir of input folder. """
        test_archive = Archive(input_dir = 'input',
                               output_dir = 'output',
                               names_sar = ['nersc_sar_primary','nersc_sar_secondary'],
                               names_amsr2 = ['amsr2'],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 50,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        test_archive.OUTPATH = ""
        with mock.patch("os.listdir", return_value=["a.nc"]):
            test_archive.get_unprocessed_files()
        self.assertEqual(test_archive.files, ['a.nc'])
        self.assertEqual(test_archive.processed_files, [])

    def test_function_update_processed_files(self):
        """processed_files attribute must be updated based on the content of 'files' and calling the
        function 'update_processed_files'."""
        temp_fold = tempfile.TemporaryDirectory()
        test_archive = Archive(input_dir = 'input',
                               output_dir = temp_fold.name,
                               names_sar = ['nersc_sar_primary','nersc_sar_secondary'],
                               names_amsr2 = ['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 'btemp_10.7h', 'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 'btemp_23.8h', 'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v', 'btemp_89.0ah', 'btemp_89.0bh', 'btemp_89.0av', 'btemp_89.0bv', 'btemp_89.0h', 'btemp_89.0v'],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 50,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        test_archive.processed_files = []
        test_archive.files = ["0.nc","1.nc","2.nc","3.nc"]
        test_archive.update_processed_files(1)
        test_archive.update_processed_files(3)
        self.assertEqual(test_archive.processed_files, ["1.nc", "3.nc"])

    def test_function_check_file_healthiness(self):
        """test unhealthiness because of lackness of 'polygon_icechart'."""
        # For this test it need :
        # test_archive.RM_SWATH = 0
        # test_archive.WINDOW_SIZE = (50, 50)
        # test_archive.AMSR_LABELS = ["btemp_6.9h"]
        test_archive = Archive(input_dir = 'input',
                               output_dir = 'output',
                               names_sar = ['nersc_sar_primary','nersc_sar_secondary'],
                               names_amsr2 = ["btemp_6.9h"],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 0,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        # the first case of unhealthy file
        fil = mock.Mock(variables=[""])
        self.assertFalse(test_archive.check_file_healthiness(fil, "fake_name"))

    def test_function_check_file_healthiness_2(self):
        """test unhealthiness because of missing AMSR file"""
        # For this test it need :
        # test_archive.RM_SWATH = 0
        # test_archive.WINDOW_SIZE = (50, 50)
        # test_archive.AMSR_LABELS = ["btemp_6.9h"]
        test_archive = Archive(input_dir = 'input',
                               output_dir = 'output',
                               names_sar = ['nersc_sar_primary','nersc_sar_secondary'],
                               names_amsr2 = ["btemp_6.9h"],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 0,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        # the second case of unhealthy file
        fil = mock.Mock(variables=['polygon_icechart'])
        self.assertFalse(test_archive.check_file_healthiness(fil, "fake_name"))

    def test_function_check_file_healthiness_3(self):
        """test unhealthiness because of small data window size"""
        # For this test it need :
        # test_archive.RM_SWATH = 0
        # test_archive.WINDOW_SIZE = (50, 50)
        # test_archive.AMSR_LABELS = ["btemp_6.9h"]
        test_archive = Archive(input_dir = 'input',
                               output_dir = 'output',
                               names_sar = ['sar1'],
                               names_amsr2 = ["btemp_6.9h"],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 0,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        # the third case of unhealthy file
        fil = mock.Mock(variables=['polygon_icechart', 'btemp_6.9h'],
                        aoi_upperleft_sample=3,
                        aoi_lowerright_sample=12,
                        aoi_lowerright_line=2,
                        aoi_upperleft_line=14,
                       )
        self.assertFalse(test_archive.check_file_healthiness(fil, "fake_name"))

    def test_function_check_file_healthiness_4(self):
        """test healthiness file for a healthy file"""
        # For this test it need :
        # test_archive.RM_SWATH = 0
        # test_archive.WINDOW_SIZE = (50, 50)
        # test_archive.AMSR_LABELS = ["btemp_6.9h"]
        test_archive = Archive(input_dir = 'input',
                               output_dir = 'output',
                               names_sar = ['sar1'],
                               names_amsr2 = ["btemp_6.9h"],
                               window_sar = 50,
                               window_amsr2 = 50,
                               stride_sar = 50,
                               stride_amsr2 = 50,
                               resample_step_amsr2 = 50,
                               resize_step_sar = 50,
                               rm_swath = 0,
                               distance_threshold = 50,
                               encoding = "one_hot_binary")
        # file is healthy
        fil = mock.Mock(variables=['polygon_icechart', 'btemp_6.9h'],
                        aoi_upperleft_sample=3,
                        aoi_lowerright_sample=120,
                        aoi_lowerright_line=200,
                        aoi_upperleft_line=14,
                       )
        self.assertTrue(test_archive.check_file_healthiness(fil, "fake_name"))

    # @mock.patch("archive.np.ma.getdata", return_value=np.array(((1, 2), (3, 4))))
    # def test_function_read_icechart_coding(self, mock_get_data):
    #     """test the reading activities of icechart and coding it"""
    #     fil = {"polygon_codes": ['id;CT;CA;SA;FA;CB;SB;FB;CC;SC;FC;CN;CD;CF;POLY_TYPE',
    #                              '33;92;-9;91; 8;-9;-9;-9;-9;-9;-9;-9;-9;-9;I',
    #                              '35;92;-9;91; 8;-9;-9;-9;-9;-9;-9;98;-9;-9;I'],
    #             # just because it is argument of 'getdata','polygon_icechart' needs to be in dict.
    #             # It does nothing in the test.
    #            "polygon_icechart": None}
    #     filename = '20180410T084537_S1B_AMSR2_'
    #     test_archive = Archive()
    #     test_archive.read_icechart_coding(fil, filename)
    #     self.assertEqual(test_archive.scene, '20180410T084537')
    #     self.assertEqual(test_archive.names_polygon_codes,
    #                      ['id', 'CT', 'CA', 'SA', 'FA', 'CB', 'SB', 'FB', 'CC', 'SC', 'FC'])
    #     np.testing.assert_equal(test_archive.polygon_ids, np.array([[1, 2], [3, 4]]))
    #     self.assertEqual(test_archive.map_id_to_variable_values, {
    #         33: ['92', '-9', '91', ' 8', '-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9', 'I'],
    #         35: ['92', '-9', '91', ' 8', '-9', '-9', '-9', '-9', '-9', '-9', '98', '-9', '-9', 'I']
    #                                                              })


    # @mock.patch("archive.np.savez")
    # def test_function_write_batches(self, mock_savez):
    #     """Test writing the correct content with a correct filename into npz files"""
    #     test_archive = Archive(input_dir = "20180410T084537",
    #                            output_dir = "/etc",
    #                            names_sar = ['sar'],
    #                            names_amsr2 = ["btemp_6.9h"],
    #                            window_sar = 50,
    #                            window_amsr2 = 50,
    #                            stride_sar = 50,
    #                            stride_amsr2 = 50,
    #                            resample_step_amsr2 = 50,
    #                            resize_step_sar = 50,
    #                            rm_swath = 0,
    #                            distance_threshold = 50,
    #                            encoding = "one_hot_binary")
    #     # test_archive.scene = "20180410T084537"
    #     # test_archive.OUTPATH = "/etc"
    #     # test_archive.NERSC = "nersc_"
    #     # test_archive.final_ful_mask = None
    #     # test_archive.final_mask_with_amsr2_size = None
    #     # test_archive.SAR_NAMES = ['sar']
    #     # test_archive.AMSR_LABELS = ['btemp_6.9h']
    #     # test_archive.names_polygon_codes = ['id', 'CT']
    #     test_archive.PROP = {'sar': [7, 8, 9],
    #                          'btemp_6_9h': [4, 5, 6],
    #                          'CT': [1, 2, 3],
    #                          '_locs': [(11, 12), (13, 14), (15, 16)]}
    #     test_archive.write_batches()
    #     self.assertEqual(mock_savez.call_args_list[0],
    #                      mock.call('/etc/20180410T084537_000000_nersc_-11_12',
    #                                sar=7, btemp_6_9h=4, CT=1, _locs=(11, 12)
    #                               )
    #                     )
    #     self.assertEqual(mock_savez.call_args_list[1],
    #                      mock.call('/etc/20180410T084537_000001_nersc_-13_14',
    #                                sar=8, btemp_6_9h=5, CT=2, _locs=(13, 14)
    #                               )
    #                     )
    #     self.assertEqual(mock_savez.call_args_list[2],
    #                      mock.call('/etc/20180410T084537_000002_nersc_-15_16',
    #                                sar=9, btemp_6_9h=6, CT=3, _locs=(15, 16)
    #                               )
    #                     )


    # @mock.patch('archive.Amsr2Batches.__init__', return_value=None)
    # @mock.patch('archive.SarBatches.__init__', return_value=None)
    # @mock.patch('archive.OutputBatches.__init__', return_value=None)
    # @mock.patch('archive.Archive.__init__', return_value=None)
    # @mock.patch("archive.view_as_windows")
    # @mock.patch("archive.Archive.check_file_healthiness")
    # @mock.patch("archive.Archive.read_icechart_coding")
    # @mock.patch("archive.Archive.calculate_mask")
    # @mock.patch("archive.Archive.calculate_batches_for_masks")
    # @mock.patch("archive.Batches.pad_and_batch")
    # @mock.patch("archive.Batches.calculate_variable_ML", side_effect=[{1: 1}, {2: 2}, {3: 3}])
    # def test_function_process_dataset(self, mock_variable_ML, mock_pad_and_batch,
    #         mock_batches_for_masks, mock_calculate_mask, mock_read, mock_check, mock_view_as_win,
    #         mock_init_Archive, mock_init_Output_ba, mock_init_Sar_ba, mock_init_Amsr2_ba):
    #     """Test that the functions are called correctly and the PROP is correctly updated"""
    #     fil = mock.Mock()
    #     filename = mock.Mock()
    #     test_archive = Archive()
    #     test_archive.PROP = {}
    #     test_archive.mask_batches_amsr2 = None
    #     test_archive.mask_batches = None
    #     test_archive.process_dataset(fil, filename)
    #     # PROP must be assembled version of output of function
    #     # which is created stepwisely with side effect
    #     self.assertEqual(test_archive.PROP, {1: 1, 2: 2, 3: 3})
    #     mock_check.assert_called_once()
    #     mock_read.assert_called_once()
    #     mock_calculate_mask.assert_called_once()
    #     mock_batches_for_masks.assert_called_once()
    #     mock_pad_and_batch.assert_called()
    #     mock_init_Output_ba.assert_called_once()
    #     mock_init_Sar_ba.assert_called_once()
    #     mock_init_Amsr2_ba.assert_called_once()
