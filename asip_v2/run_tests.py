import unittest

from tests.test_archive import (BatchesTestCases, SarBatchesTestCases,
                                OutputBatchesTestCases, DistanceBatchesTestCases, Amsr2BatchesTestCases,
                                ArchiveTestCases)

from tests.test_data_generator import (DataGeneratorTestCases, HugoDataGeneratorTestCases, DataGenerator_sod_fTest_case)

if __name__ == '__main__':
    unittest.main(failfast=True, buffer=True)
