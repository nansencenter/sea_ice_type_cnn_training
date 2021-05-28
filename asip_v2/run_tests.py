import unittest

from tests.test_utility import UtilityFunctionsTestCases, ConfigureTestCases
from tests.test_build_dataset import BuildDatasetTestCases
from tests.test_train_model import FileBasedConfigureTestCases
from tests.test_apply_model import MemoryBasedConfigureTestCases

#from tests.test_data_generator import DataGeneratorTestCases
#from tests.test_archive import ArchiveTestCases



if __name__ == '__main__':
    unittest.main(failfast=True)
