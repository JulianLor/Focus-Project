import unittest
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from src.main.python.Electromagnet import Electromagnet
from src.main.python.Helper_functions import Biot_Savart_Law

class TestElectromagnet(unittest.TestCase):

    def setUp(self):
        pass

    # testing whether the returned distance is correct
    def test_varying_distance_calc_r(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.varying_distance_calc(5, 10, 1, 'r')

        # check result with predicted result
        prediction = 0.755
        self.assertAlmostEqual(result, prediction)

    # testing whether the returned distance is correct
    def test_varying_distance_calc_d(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.varying_distance_calc(5, 10, 1, 'd')

        # check result with predicted result
        prediction = 0.76
        self.assertAlmostEqual(result, prediction)

    # testing if the points are rotated correctly ('normal')
    def test_rotating_point_to_pos_normal(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.rotating_point_to_pos(np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2]), 'normal')

        # check result with predicted result
        prediction = np.array([np.cos(np.pi / 12), 0, np.sin(np.pi / 12)])
        assert_array_almost_equal(result, prediction)

    # testing if the points are rotated correctly ('inverse')
    def test_rotating_point_to_pos_inverse(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.rotating_point_to_pos(np.array([np.cos(np.pi / 12), 0, np.sin(np.pi / 12)]), 'inverse')

        # check result with predicted result
        prediction = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
        assert_array_almost_equal(result, prediction)

    # testing if the representative length of Electromagnet point is correct
    def test_get_dl(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_dl(np.array([np.cos(np.pi / 12), 0, np.sin(np.pi / 12)]))

        # check result with predicted result
        prediction = np.sqrt(2) * np.pi / 50
        self.assertAlmostEqual(result, prediction)

    # testing if the Electromagnet point is being returned correctly
    def test_get_solenoid_point(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_solenoid_point(np.array([0.02, np.pi / 2, 0.15]))

        # check result with predicted result
        prediction = list(test.rotating_point_to_pos(np.array([0, 0.02, 0.15]), 'normal'))
        assert_array_almost_equal(result, prediction)

    # testing if the current vector is being returned correctly
    def test_get_current_vector(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_current_vector(np.pi / 6)

        # check result with predicted result
        prediction = list(test.rotating_point_to_pos(np.array([-1 / 2, np.sqrt(3) / 2, 0]), 'normal'))
        assert_array_almost_equal(result, prediction)

    # testing if the point parameters are returned correctly
    def test_get_point_parameters(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_parameters(5, 5, 5)

        # check result with predicted result
        prediction = [test.varying_distance_calc(test.d_c, test.get_n_d(), 5, 'r'),
                      np.pi / 5,
                      test.varying_distance_calc(test.l_c, test.get_n_l(), 5, 'd') + test.d_z]
        assert_array_almost_equal(np.array(result), np.array(prediction))

    # testing if the looping over the electromagnet works
    def test_get_solenoid_point_coords_shape(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_solenoid_point_coords()

        # check result with predicted result
        prediction = np.zeros((test.get_total_points(), 3))
        assert_array_almost_equal(result.shape, prediction.shape)

    # testing if the B-flux is correctly calculated for single point of electromagnet
    def test_get_B_flux(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', 0], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_B_flux(np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([0, 1, 0]))

        # check result with predicted result
        prediction = test.DELTA * Biot_Savart_Law(np.array([0,0,-1]), np.array([0,1,0]), 2 * np.pi / test.POINTS_PER_TURN)
        assert_array_almost_equal(result.shape, prediction.shape)

    # testing if the B-flux is correctly calculated for the entire electromagnet
    def test_get_solenoid_B_flux(self):
        # setup test instance
        test = Electromagnet(0, ['y', 0], 1, 0.5, 0.01)
        # set the test-electromagnets current to 1 A
        test.set_solenoid_current(1)
        # get the result that is being verified
        result = test.get_solenoid_B_flux(np.array([0,0,0]))

        # check result with predicted result
        prediction = test.POINTS_PER_TURN * test.get_B_flux(np.array([0,0,0]), np.array([0.5,0,0]), np.array([0,1,0]))
        assert_array_almost_equal(result.shape, prediction.shape)
