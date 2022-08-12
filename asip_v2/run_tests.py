import unittest
import unittest

from tests.test_archive import (BatchesTestCases, SarBatchesTestCases,
                                OutputBatchesTestCases, DistanceBatchesTestCases, Amsr2BatchesTestCases,
                                ArchiveTestCases)

from tests.test_data_generator import (DataGeneratorTestCases, HugoDataGeneratorTestCases, HugoBinaryGeneratorTestCases, DataGenerator_sod_fTest_case)
from tests.test_utility import (UtilityFunctionsTestCases, ConfigureTestCases)
from tests.test_build_dataset import BuildDatasetTestCases

if __name__ == '__main__':
    unittest.main(failfast=True, buffer=True)