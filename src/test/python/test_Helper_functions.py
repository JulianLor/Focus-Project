import unittest
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from src.main.python.Helper_functions import rotate_vector

class TestHelperFunctions(unittest.TestCase):

    def setUp(self):
        pass

    # testing if the vectors are rotated correctly
    def test_rotate_vector(self):
        # get the results that is being verified
        result_1 = rotate_vector(np.array([1, 0, 0]), 'y', np.pi / 2)
        result_2 = rotate_vector(np.array([0, 1, 0]), 'x', np.pi / 2)
        result_3 = rotate_vector(np.array([1, 0, 0]), 'z', np.pi / 2)

        # check results with predicted results
        prediction_1 = np.array([0, 0, -1])
        prediction_2 = np.array([0, 0, 1])
        prediction_3 = np.array([0, 1, 0])
        assert_array_almost_equal(result_1, prediction_1)
        assert_array_almost_equal(result_2, prediction_2)
        assert_array_almost_equal(result_3, prediction_3)

    # testing if the vectors are rotated correctly
    def test_rotate_vector_2(self):
        # get the results that is being verified
        result_1 = rotate_vector(np.array([1, 1, 0]), 'y', np.pi / 2)
        result_2 = rotate_vector(np.array([1, 1, 0]), 'x', np.pi / 2)
        result_3 = rotate_vector(np.array([1, 0, 1]), 'z', np.pi / 2)

        # check results with predicted results
        prediction_1 = np.array([0, 1, -1])
        prediction_2 = np.array([1, 0, 1])
        prediction_3 = np.array([0, 1, 1])
        assert_array_almost_equal(result_1, prediction_1)
        assert_array_almost_equal(result_2, prediction_2)
        assert_array_almost_equal(result_3, prediction_3)

    # testing if the vectors are rotated correctly
    def test_rotate_vector_3(self):
        # get the results that is being verified
        result_1 = rotate_vector(np.array([1, 1, 0]), 'y', np.pi / 4)
        result_2 = rotate_vector(np.array([1, 1, 0]), 'x', np.pi / 4)
        result_3 = rotate_vector(np.array([1, 0, 1]), 'z', np.pi / 4)

        # check results with predicted results
        prediction_1 = np.array([np.sqrt(2)/2, 1, -np.sqrt(2)/2])
        prediction_2 = np.array([1, np.sqrt(2)/2, np.sqrt(2)/2])
        prediction_3 = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 1])
        assert_array_almost_equal(result_1, prediction_1)
        assert_array_almost_equal(result_2, prediction_2)
        assert_array_almost_equal(result_3, prediction_3)