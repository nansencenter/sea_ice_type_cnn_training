import unittest

from tests.data_generator_tests import DataGeneratorTestCases
from tests.utility_tests import (InitializationTestCases, ConfigureTestCases,
                                FileBasedConfigureTestCases, MemoryBasedConfigureTestCases)
from tests.archive_tests import ArchiveTestCases

if __name__ == '__main__':
    unittest.main(failfast=True)
