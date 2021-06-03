import tempfile
import unittest
import unittest.mock as mock

import numpy as np
from archive import Amsr2Batches, Archive, Batches, OutputBatches, SarBatches
from skimage.util.shape import view_as_windows


class BatchesTestCases(unittest.TestCase):
    """tests for batches class"""

    @mock.patch("archive.view_as_windows")
    def test_function_view_as_windows(self, mock_view_as_window):
        """view_as_windows shall be called correctly"""
        test_batch = Batches()
        test_batch.WINDOW_SIZE = (2,2)
        test_batch.STRIDE = 4
        test_batch.view_as_windows(1)
        mock_view_as_window.assert_called_with(1, (2, 2), 4)

    def test_function_convert(self):
        """first argument must be delivered with 'convert' function"""
        test_batch = Batches()
        self.assertEqual(test_batch.convert(1,2), 1)

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

    def test_function_calculate_pading(self):
        """test calculate pading with correct dtype and constant value """
        test_batch = Batches()
        test_batch.pads = [1, 2, 3, 4]
        array = np.array([[1, 2], [3, 4]])
        ans = test_batch.calculate_pading(array,np.float32, 100)
        np.testing.assert_equal(ans,np.array([[100., 100., 100., 100., 100., 100., 100., 100., 100.],
                                              [100., 100., 100.,   1.,   2., 100., 100., 100., 100.],
                                              [100., 100., 100.,   3.,   4., 100., 100., 100., 100.],
                                              [100., 100., 100., 100., 100., 100., 100., 100., 100.],
                                              [100., 100., 100., 100., 100., 100., 100., 100., 100.]],
                                              dtype = np.float32))

    @mock.patch("archive.Batches.convert", return_value = np.array(3))
    @mock.patch("archive.Batches.pading", return_value = "return_value_of_pading")
    @mock.patch("numpy.ma.getdata", return_value = "return_value_of_get_data")
    def test_function_pad_and_batch(self, mock_get_data, mock_pading, mock_convert):
        """self.batches_array must be set properly"""
        test_batch = Batches()
        test_batch.WINDOW_SIZE = 1
        test_batch.STRIDE = 1
        test_batch.loop_list = ["fake_name"]
        fil = {"fake_name": "fake_value"}
        test_batch.pad_and_batch(fil)
        mock_get_data.assert_called_with('fake_value')
        mock_pading.assert_called_with("return_value_of_get_data")
        mock_convert.assert_called_with('return_value_of_pading', 'fake_name')
        self.assertEqual(test_batch.batches_array, {'fake_name': np.array(3)})


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

    @mock.patch("archive.SarBatches.calculate_pading")
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_pading(self, mock_archive, mock_pading):
        """
        calculate_pading function must be called with correct astype and constant_value for output
        """
        test_batch = SarBatches(archive_=mock_archive)
        test_batch.pading('')
        mock_pading.assert_called_once_with('', np.float32, None)


class OutputBatchesTestCases(unittest.TestCase):
    """tests for OutputBatches class"""
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_conventer(self, mock_archive):
        """name_conventer should return the correct names"""
        test_batch = OutputBatches(archive_=mock_archive)
        test_batch.names_polygon_codes = ['id', 'CT', 'CA', 'SA', 'FA', 'CB']
        #'id' should not be included in the output of function "name_conventer"
        self.assertEqual(test_batch.name_conventer(0), 'CT')
        self.assertEqual(test_batch.name_conventer(2), 'SA')

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_for_getdata(self, mock_archive):
        """name_for_getdata should always return a constant value"""
        test_batch = OutputBatches(archive_=mock_archive)
        self.assertEqual(test_batch.name_for_getdata(""), "polygon_icechart")

    @mock.patch("archive.OutputBatches.calculate_pading")
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_pading(self, mock_archive, mock_pading):
        """
        calculate_pading function must be called with correct astype and constant_value for output
        """
        test_batch = OutputBatches(archive_=mock_archive)
        test_batch.pading('')
        mock_pading.assert_called_once_with('', np.byte, 0)

    @mock.patch("archive.OutputBatches.encode_icechart")
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_convert(self, mock_archive, mock_encode):
        """encode_icechart function must be called with correct arguments"""
        test_batch = OutputBatches(archive_=mock_archive)
        test_batch.convert('1', "2")
        mock_encode.assert_called_once_with('1', "2")

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_encode_icechart(self, mock_archive):
        """icechart must be encoded with map_id_to_variable_values and the element that is provided"""
        test_batch = OutputBatches(archive_=mock_archive)
        test_batch.map_id_to_variable_values = {33: ['92', '-9', '91', '8', '-8'],
                                                45: ['30', '10', '95', '3', '11']}
        original_array = np.array([[45, 33], [33, 45]])
        np.testing.assert_equal(
                                test_batch.encode_icechart(original_array, 0),
                                np.array([[30, 92], [92, 30]])
                               )
        np.testing.assert_equal(
                                test_batch.encode_icechart(original_array, 1),
                                np.array([[10, -9], [-9, 10]])
                               )

    @mock.patch("archive.Batches.resize")
    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_resize(self, mock_archive, mock_resize):
        """resize of the parent class must be called"""
        test_batch = OutputBatches(archive_=mock_archive)
        test_batch.resize('')
        mock_resize.assert_called_once()


class Amsr2BatchesTestCases(unittest.TestCase):
    """tests for Amsr2Batches class"""

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_calculate_variable_ML(self, mock_archive):
        """The return value of the function must be set properly because PROP will be updated based
        on it.This function filters outs the values of 'batches_array' based on 'batches_mask'"""
        test_batch = Amsr2Batches(archive_=mock_archive)
        test_batch.WINDOW_SIZE = 1
        test_batch.STRIDE = 1
        test_batch.step = 1
        test_batch.astype = np.float32
        test_batch.loop_list = ["fake_name"]
        test_batch.batches_array = {"fake_name": test_batch.view_as_windows(
                                                                        np.arange(4).reshape(2, 2))}
        test_batch.batches_mask = np.array(((False, True), (True, False)))
        ans = test_batch.calculate_variable_ML()
        self.assertEqual(ans,
                         # two middle Trues in mask prevent (0,1) and (1,0)
                         {'_locs':     [(0, 0), (1, 1)],
                          'fake_name': [np.array([[0.]], dtype=np.float32),
                                        np.array([[3.]], dtype=np.float32)
                                       ]
                         }
                        )

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_pading(self, mock_archive):
        """pading should return the same array without any padding action"""
        test_batch = Amsr2Batches(archive_=mock_archive)
        np.testing.assert_equal(test_batch.pading(np.array([[0.]])), np.array([[0.]]))

    @mock.patch('utility.Archive.__init__', return_value=None)
    def test_function_name_conventer(self, mock_archive):
        """name_conventer should convert the name"""
        test_batch = Amsr2Batches(archive_=mock_archive)
        self.assertEqual(test_batch.name_conventer("btemp_89.0h"), "btemp_89_0h")


class ArchiveTestCases(unittest.TestCase):
    """tests for Archive class"""

    def test_function_get_unprocessed_files(self):
        """ Without having a file named 'processed_files.json' in the output folder,
        'files' attribute must be filled with what comes out of litsdir of input folder. """
        test_archive = Archive()
        test_archive.OUTPATH = ""
        with mock.patch("os.listdir", return_value=["a.nc"]):
            test_archive.get_unprocessed_files()
        self.assertEqual(test_archive.files, ['a.nc'])
        self.assertEqual(test_archive.processed_files, [])

    def test_function_update_processed_files(self):
        """processed_files attribute must be updated based on the content of 'files' and calling the
        function 'update_processed_files'."""
        test_archive = Archive()
        test_archive.processed_files = []
        test_archive.files = ["0.nc","1.nc","2.nc","3.nc"]
        temp_fold = tempfile.TemporaryDirectory()
        test_archive.OUTPATH = temp_fold.name
        test_archive.update_processed_files(1)
        test_archive.update_processed_files(3)
        self.assertEqual(test_archive.processed_files, ["1.nc", "3.nc"])

    def test_function_check_file_healthiness(self):
        """test unhealthiness because of lackness of 'polygon_icechart'."""
        test_archive = Archive()
        test_archive.RM_SWATH = 0
        test_archive.WINDOW_SIZE = (50, 50)
        # the first case of unhealthy file
        fil = mock.Mock(variables=[""])
        test_archive.AMSR_LABELS = ["btemp_6.9h"]
        self.assertFalse(test_archive.check_file_healthiness(fil, "fake_name"))

    def test_function_check_file_healthiness_2(self):
        """test unhealthiness because of missing AMSR file"""
        test_archive = Archive()
        test_archive.RM_SWATH = 0
        test_archive.WINDOW_SIZE = (50, 50)
        # the second case of unhealthy file
        fil = mock.Mock(variables=['polygon_icechart'])
        test_archive.AMSR_LABELS = ["btemp_6.9h"]
        self.assertFalse(test_archive.check_file_healthiness(fil, "fake_name"))

    def test_function_check_file_healthiness_3(self):
        """test unhealthiness because of small data window size"""
        test_archive = Archive()
        test_archive.RM_SWATH = 0
        test_archive.WINDOW_SIZE = (50, 50)
        test_archive.AMSR_LABELS = ["btemp_6.9h"]
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
        test_archive = Archive()
        test_archive.RM_SWATH = 0
        test_archive.WINDOW_SIZE = (50, 50)
        test_archive.AMSR_LABELS = ["btemp_6.9h"]
        # file is healthy
        fil = mock.Mock(variables=['polygon_icechart', 'btemp_6.9h'],
                        aoi_upperleft_sample=3,
                        aoi_lowerright_sample=120,
                        aoi_lowerright_line=200,
                        aoi_upperleft_line=14,
                       )
        self.assertTrue(test_archive.check_file_healthiness(fil, "fake_name"))

    @mock.patch("archive.np.ma.getdata", return_value=np.array(((1, 2), (3, 4))))
    def test_function_read_icechart_coding(self, mock_get_data):
        """test the reading activities of icechart and coding it"""
        fil = {"polygon_codes": ['id;CT;CA;SA;FA;CB;SB;FB;CC;SC;FC;CN;CD;CF;POLY_TYPE',
                                 '33;92;-9;91; 8;-9;-9;-9;-9;-9;-9;-9;-9;-9;I',
                                 '35;92;-9;91; 8;-9;-9;-9;-9;-9;-9;98;-9;-9;I'],
                # just because it is argument of 'getdata','polygon_icechart' needs to be in dict.
                # It does nothing in the test.
               "polygon_icechart": None}
        filename = '20180410T084537_S1B_AMSR2_'
        test_archive = Archive()
        test_archive.read_icechart_coding(fil, filename)
        self.assertEqual(test_archive.scene, '20180410T084537')
        self.assertEqual(test_archive.names_polygon_codes,
                         ['id', 'CT', 'CA', 'SA', 'FA', 'CB', 'SB', 'FB', 'CC', 'SC', 'FC'])
        np.testing.assert_equal(test_archive.polygon_ids, np.array([[1, 2], [3, 4]]))
        self.assertEqual(test_archive.map_id_to_variable_values, {
            33: ['92', '-9', '91', ' 8', '-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9', 'I'],
            35: ['92', '-9', '91', ' 8', '-9', '-9', '-9', '-9', '-9', '-9', '98', '-9', '-9', 'I']
                                                                 })

    def test_function_get_the_mask_of_sar_size_data_with_distance_threshold(self):
        """test the assimilation of masks from different locations.in this test, the upper part of
        array is masked because of distance_thershold."""
        sar_names = ['fake_sar_name']
        fil = {'fake_sar_name': np.array([[999, 998], [997, 996]]),
               'distance_map': np.array([[10, 11], [32, 33]]),
               'polygon_icechart': np.array([[1, 2], [3, 4]])}
        np.testing.assert_equal(Archive.get_the_mask_of_sar_size_data(sar_names, fil, 20),
                                # 10 & 11 are below 20, and 32 & 33 are above it
                                np.array([[True, True], [False, False]]))

    def test_function_get_the_mask_of_sar_size_data_with_none_and_masked_locations(self):
        """test the assimilation of masks from different locations. In this test, each quarter of
        array is masked by fil.The first row is masked by "sar". The second row is masked by
        'polygon_icechart'. Finally the assembled mask must be completely true(enable).
        'distance_map' does nothing in this test."""
        sar_names = ["sar"]
        fil = {"sar": np.ma.masked_equal([[3, None], [1, 2]], 3).astype(float),
               'distance_map': np.array([[10, 11], [12, 13]]),
               'polygon_icechart': np.ma.masked_equal([[1, 2], [None, 3]], 3).astype(float)}
        np.testing.assert_equal(Archive.get_the_mask_of_sar_size_data(sar_names, fil, 1),
                                np.array([[True, True], [True, True]]))

    def test_function_get_the_mask_of_amsr2_data(self):
        """test the mask creation for amsr2 data."""
        test_archive = Archive()
        amsr_labels = [""]
        fil = {"": np.ma.masked_equal([[1, 2], [None, 3]], 3).astype(float)}
        test_archive.ASPECT_RATIO = 2
        mask_amsr, shape_mask_amsr_0, shape_mask_amsr_1 = test_archive.get_the_mask_of_amsr2_data(
                                                                                    amsr_labels, fil)
        # the answer must be double sized because "ASPECT_RATIO = 2"
        np.testing.assert_equal(mask_amsr, np.array([[False, False, False, False],
                                                     [False, False, False, False],
                                                     [ True,  True,  True,  True],
                                                     [ True,  True,  True,  True]]))
        # size must be the size of what is inside the "fil"
        self.assertEqual(shape_mask_amsr_0, 2)
        self.assertEqual(shape_mask_amsr_1, 2)

    def test_function_pad_the_mask_of_sar_based_on_size_amsr(self):
        """test the correct behaviour of padding"""
        mask_sar_size = np.arange(6,12).reshape(2, 3)
        # mask_sar_size = array([[ 6,  7,  8],
        #                        [ 9, 10, 11]])
        mask_amsr = np.arange(36).reshape(6, 6)
        # mask_amsr = array([[ 0,  1,  2,  3,  4,  5],
        #                    [ 6,  7,  8,  9, 10, 11],
        #                    [12, 13, 14, 15, 16, 17],
        #                    [18, 19, 20, 21, 22, 23],
        #                    [24, 25, 26, 27, 28, 29]])
        mask_sar_size, pads = Archive.pad_the_mask_of_sar_based_on_size_amsr(mask_amsr, mask_sar_size)
        np.testing.assert_equal(mask_sar_size, np.array([[ 1,  1,  1,  1,  1,  1],
                                                         [ 1,  1,  1,  1,  1,  1],
                                                         [ 1,  6,  7,  8,  1,  1],
                                                         [ 1,  9, 10, 11,  1,  1],
                                                         [ 1,  1,  1,  1,  1,  1],
                                                         [ 1,  1,  1,  1,  1,  1]]))
        self.assertEqual(pads, (2, 2, 1, 2))
        mask_sar_size = np.arange(6,12).reshape(3, 2)
        # mask_sar_size = array([[ 6,  7],
        #                        [ 8, 9],
        #                        [10, 11]])
        mask_sar_size, pads = Archive.pad_the_mask_of_sar_based_on_size_amsr(mask_amsr, mask_sar_size)
        np.testing.assert_equal(mask_sar_size, np.array([[ 1,  1,  1,  1,  1,  1],
                                                         [ 1,  1,  6,  7,  1,  1],
                                                         [ 1,  1,  8,  9,  1,  1],
                                                         [ 1,  1, 10, 11,  1,  1],
                                                         [ 1,  1,  1,  1,  1,  1],
                                                         [ 1,  1,  1,  1,  1,  1]]))
        self.assertEqual(pads, (1, 2, 2, 2))



    def test_function_downsample_mask_for_amsr2(self):
        """
        downsampling should be done correctly. Even one of elements inside the patch is True, then
        the whole patch must be masked in the downsampled image
        """
        final_ful_mask = np.arange(36).reshape(6, 6)>18
        #array([[False, False, False, False, False, False],
        #       [False, False, False, False, False, False],
        #       [False, False, False, False, False, False],
        #       [False,  True,  True,  True,  True,  True],
        #       [ True,  True,  True,  True,  True,  True],
        #       [ True,  True,  True,  True,  True,  True]])
        # middle row in the downsized matrix should have true values because part of it are masked
        np.testing.assert_equal(Archive.downsample_mask_for_amsr2(final_ful_mask, 3, 3),
                                                                  np.array([[False, False, False],
                                                                            [ True,  True,  True],
                                                                            [ True,  True,  True]]))

    @mock.patch("archive.Archive.get_the_mask_of_sar_size_data", return_value="sarout")
    @mock.patch("archive.Archive.get_the_mask_of_amsr2_data", return_value=['aout0','aout1','aout2'])
    @mock.patch("archive.Archive.pad_the_mask_of_sar_based_on_size_amsr", return_value=['rp1','rp2'])
    @mock.patch("archive.np.ma.mask_or", return_value=np.array([[1]]))
    @mock.patch("archive.Archive.downsample_mask_for_amsr2", return_value=np.array([[1,2]]))
    @mock.patch("archive.np.full")
    def test_function_calculate_mask(self, mock_full, mock_down, mock_or, mock_pad, mock_ams, mock_sar):
        """test the calling of methods correctly in this function"""
        fil = mock.Mock()
        test_archive = Archive()
        test_archive.apply_instead_of_training = True
        test_archive.calculate_mask(fil)
        mock_ams.assert_called()
        mock_sar.assert_called()
        mock_or.assert_called_with('rp1', 'aout0')
        mock_pad.assert_called_with('aout0', 'sarout')
        mock_down.assert_called_with(np.array([[1]]), 'aout1', 'aout2')
        mock_full.assert_called()

    @mock.patch("archive.np.savez")
    def test_function_write_batches(self, mock_savez):
        """Test writing the correct content with a correct filename into npz files"""
        test_archive = Archive()
        test_archive.scene = "20180410T084537"
        test_archive.OUTPATH = "/etc"
        test_archive.NERSC = "nersc_"
        test_archive.final_ful_mask = None
        test_archive.final_mask_with_amsr2_size = None
        test_archive.SAR_NAMES = ['sar']
        test_archive.AMSR_LABELS = ['btemp_6.9h']
        test_archive.names_polygon_codes = ['id', 'CT']
        test_archive.PROP = {'sar': [7, 8, 9],
                             'btemp_6_9h': [4, 5, 6],
                             'CT': [1, 2, 3],
                             '_locs': [(11, 12), (13, 14), (15, 16)]}
        test_archive.write_batches()
        self.assertEqual(mock_savez.call_args_list[0],
                         mock.call('/etc/20180410T084537_000000_nersc_-11_12',
                                   sar=7, btemp_6_9h=4, CT=1, _locs=(11, 12)
                                  )
                        )
        self.assertEqual(mock_savez.call_args_list[1],
                         mock.call('/etc/20180410T084537_000001_nersc_-13_14',
                                   sar=8, btemp_6_9h=5, CT=2, _locs=(13, 14)
                                  )
                        )
        self.assertEqual(mock_savez.call_args_list[2],
                         mock.call('/etc/20180410T084537_000002_nersc_-15_16',
                                   sar=9, btemp_6_9h=6, CT=3, _locs=(15, 16)
                                  )
                        )

    @mock.patch("archive.view_as_windows")
    def test_function_calculate_batches_for_masks(self, mock_view_as_win):
        """
        Test that self.mask_batches aand self.mask_batches_amsr2 are correctly created by correct
        call of view_as_windows function
        """
        test_archive = Archive()
        test_archive.final_ful_mask = 1
        test_archive.WINDOW_SIZE = (2, 2)
        test_archive.STRIDE_SAR_SIZE = 3
        test_archive.final_mask_with_amsr2_size = 4
        test_archive.WINDOW_SIZE_AMSR2 = (5, 5)
        test_archive.STRIDE_AMS2_SIZE = 6
        test_archive.calculate_batches_for_masks()
        self.assertEqual(mock_view_as_win.call_args_list[0], mock.call(1, (2, 2), 3))
        self.assertEqual(mock_view_as_win.call_args_list[1], mock.call(4, (5, 5), 6))

    @mock.patch('archive.Amsr2Batches.__init__', return_value=None)
    @mock.patch('archive.SarBatches.__init__', return_value=None)
    @mock.patch('archive.OutputBatches.__init__', return_value=None)
    @mock.patch('archive.Archive.__init__', return_value=None)
    @mock.patch("archive.view_as_windows")
    @mock.patch("archive.Archive.check_file_healthiness")
    @mock.patch("archive.Archive.read_icechart_coding")
    @mock.patch("archive.Archive.calculate_mask")
    @mock.patch("archive.Archive.calculate_batches_for_masks")
    @mock.patch("archive.Batches.pad_and_batch")
    @mock.patch("archive.Batches.calculate_variable_ML", side_effect=[{1: 1}, {2: 2}, {3: 3}])
    def test_function_process_dataset(self, mock_variable_ML, mock_pad_and_batch,
            mock_batches_for_masks, mock_calculate_mask, mock_read, mock_check, mock_view_as_win,
            mock_init_Archive, mock_init_Output_ba, mock_init_Sar_ba, mock_init_Amsr2_ba):
        """Test that the functions are called correctly and the PROP is correctly updated"""
        fil = mock.Mock()
        filename = mock.Mock()
        test_archive = Archive()
        test_archive.PROP = {}
        test_archive.mask_batches_amsr2 = None
        test_archive.mask_batches = None
        test_archive.process_dataset(fil, filename)
        # PROP must be assembled version of output of function
        # which is created stepwisely with side effect
        self.assertEqual(test_archive.PROP, {1: 1, 2: 2, 3: 3})
        mock_check.assert_called_once()
        mock_read.assert_called_once()
        mock_calculate_mask.assert_called_once()
        mock_batches_for_masks.assert_called_once()
        mock_pad_and_batch.assert_called()
        mock_init_Output_ba.assert_called_once()
        mock_init_Sar_ba.assert_called_once()
        mock_init_Amsr2_ba.assert_called_once()
