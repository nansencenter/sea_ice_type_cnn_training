import unittest

from tests.test_apply_model import MemoryBasedConfigureTestCases
from tests.test_archive import (Amsr2BatchesTestCases, ArchiveTestCases,
                                BatchesTestCases, OutputBatchesTestCases,
                                SarBatchesTestCases)
from tests.test_build_dataset import BuildDatasetTestCases
from tests.test_data_generator import (DataGeneratorFrom_npz_FileTestCases,
                                       DataGeneratorFromMemoryTestCases,
                                       DataGeneratorTestCases)
from tests.test_train_model import FileBasedConfigureTestCases
from tests.test_utility import ConfigureTestCases, UtilityFunctionsTestCases

if __name__ == '__main__':
    unittest.main(failfast=True, buffer=True)
