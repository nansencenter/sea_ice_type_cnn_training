import unittest

from tests.test_archive import (BatchesTestCases, SarBatchesTestCases,
                                OutputBatchesTestCases, DistanceBatchesTestCases, Amsr2BatchesTestCases,
                                ArchiveTestCases)

if __name__ == '__main__':
    unittest.main(failfast=True, buffer=True)
