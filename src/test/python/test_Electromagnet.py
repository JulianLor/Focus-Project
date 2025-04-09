import unittest
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from src.main.python.Electromagnet import Electromagnet
from src.main.python.Helper_functions import Biot_Savart_Law, Lorentz_force

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

    # testing if the system exit is triggered for rotating_point_to_pos function
    def test_rotating_point_to_pos_2(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)

        # check if system exit is triggered
        with self.assertRaises(SystemExit) as cm: # Should trigger sys.exit()
            test.rotating_point_to_pos(np.array([0, 0]), 'test')

        # Check exit code
        self.assertEqual(cm.exception.code, 'Invalid mode')

    # testing if the representative length of Electromagnet point is correct
    def test_get_dl(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', 0], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_dl(0.1)

        # check result with predicted result
        prediction = 2 * np.pi / 500
        self.assertAlmostEqual(result, prediction)

    # testing if the Electromagnet point params is being returned correctly
    def test_get_point_params(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_params(0, 0, 0)

        # check result with predicted result
        r = 0.005 + (test.d_c / test.n_d) / 2
        d = 0.01 + (test.l_c / test.n_l) / 2
        theta = 0
        prediction = np.array([test.get_dl(r), r, d, theta])
        assert_array_almost_equal(result, prediction)

    # testing if the Electromagnet point params is being returned correctly
    def test_get_point_params_2(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 6], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_params(4, 2, 10)

        # check result with predicted result
        r = 0.005 + (test.d_c / test.n_d) * 4.5
        d = 0.01 + (test.l_c / test.n_l) * 2.5
        theta = (10 / test.POINTS_PER_TURN) * 2 * np.pi
        prediction = np.array([test.get_dl(r), r, d, theta])
        assert_array_almost_equal(result, prediction)

    # testing if the point coordinates are returned correctly
    def test_get_point_coords(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', 0], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_coords(np.sqrt(2), np.pi / 4, 0.5)

        # check result with predicted result
        prediction = np.array([1, 1, 0.5])
        assert_array_almost_equal(result, prediction)

    # testing if the point coordinates are returned correctly
    def test_get_point_coords_2(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 2], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_coords(1, 0, 0.5)

        # check result with predicted result
        prediction = np.array([0.5, 0, -1])
        assert_array_almost_equal(result, prediction)

    # testing if the point coordinates are returned correctly
    def test_get_point_coords_3(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 4], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_point_coords(1, np.pi/2, np.sqrt(2))

        # check result with predicted result
        prediction = np.array([1, 1, 1])
        assert_array_almost_equal(result, prediction)

    # testing if the current vector is being returned correctly
    def test_get_current_vector(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 4], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_current_vector(0)

        # check result with predicted result
        prediction = np.array([0,1,0])
        assert_array_almost_equal(result, prediction)

    # testing if the looping over the electromagnet works
    def test_get_current_vector_2(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', 0], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_current_vector(np.pi / 4)

        # check result with predicted result
        prediction = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
        assert_array_almost_equal(result, prediction)

    # testing if the looping over the electromagnet works
    def test_get_current_vector_3(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi/4], 350, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_current_vector(np.pi / 4)

        # check result with predicted result
        prediction = np.array([-1/2, np.sqrt(2) / 2, 1/2])
        assert_array_almost_equal(result, prediction)

    # testing if all point parameters of all the electromagnet's point are returned correctly
    def test_get_solenoid_params(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', np.pi / 4], 1, 0.005, 0.01)
        # get the result that is being verified
        result = test.get_solenoid_params()

        # generate prediction
        prediction = np.zeros((test.POINTS_PER_TURN, 4))
        for point in range(test.POINTS_PER_TURN):
            prediction[point] = test.get_point_params(0, 0, point)
        # check result with predicted result
        assert_array_almost_equal(result, prediction)

    # testing if the B-flux is correctly calculated for single point of electromagnet
    def test_get_B_flux(self):
        # setup test instance
        test = Electromagnet(0.01, ['y', 0], 350, 0.005, 0.01)
        test.save_solenoid_all()
        # set the test-electromagnets current to 1 A
        test.set_solenoid_current(1)
        # get the result that is being verified
        result = test.get_B_flux(np.array([1, 0, 0]), 0)

        # check result with predicted result
        prediction = test.DELTA * Biot_Savart_Law(np.array([1,0,0]), np.array([0,1,0]), 0.0007121830803688726, 1)
        assert_array_almost_equal(result, prediction)

    # testing if the system exit is triggered for get_solenoid_B_flux function
    def test_get_B_flux_2(self):
        # setup test instance
        test = Electromagnet(0, ['y', 0], 1, 0.5, 0.01)

        # check if system exit is triggered
        with self.assertRaises(SystemExit) as cm:
            test.get_B_flux(np.array([0, 0, 0]), 0)  # Should trigger sys.exit()

        # Check exit code
        self.assertEqual(cm.exception.code, 'No data defined at index: 0')

    # testing if the system exit is triggered for get_solenoid_B_flux function
    def test_get_B_flux_3(self):
        # setup test instance
        test = Electromagnet(0, ['y', 0], 1, 0.5, 0.01)
        test.save_solenoid_point_param()

        # check if system exit is triggered
        with self.assertRaises(SystemExit) as cm:
            test.get_B_flux(np.array([0, 0, 0]), 0)  # Should trigger sys.exit()

        # Check exit code
        self.assertEqual(cm.exception.code, 'No current vectors or coords at index: 0')


    # testing if the B-flux is correctly calculated for the entire electromagnet
    def test_get_solenoid_B_flux(self):
        # setup test instance
        test = Electromagnet(0, ['y', 0], 1, 0.5, 0.01)
        test.save_solenoid_all()
        # set the test-electromagnets current to 1 A
        test.set_solenoid_current(1)
        # get the result that is being verified
        result = test.get_solenoid_B_flux(np.array([0,0,0]))

        # check result with predicted result
        prediction = test.POINTS_PER_TURN * test.get_B_flux(np.array([0,0,0]), 0)
        assert_array_almost_equal(result, prediction)

    # testing if the B-flux is correctly calculated for the entire electromagnet with significant parameters
    def test_get_solenoid_B_flux_2(self):
        # setup test instance
        test = Electromagnet(0, ['y', 0], 400, 0.05, 0.03)
        test.save_solenoid_all()
        # set the test-electromagnets current to 5.5 A
        test.set_solenoid_current(5.5)
        # get the result that is being verified
        result = test.get_solenoid_B_flux(np.array([0, 0, -0.14]))

        # check result with predicted result
        prediction = np.array([0,0,1.059787e-03])
        assert_array_almost_equal(result, prediction)

    # testing if the B-flux is correctly calculated for the entire electromagnet with significant parameters
    def test_get_solenoid_B_flux_3(self):
        # setup test instance
        test = Electromagnet(0.14, ['y', np.pi/4], 400, 0.05, 0.03)
        test.save_solenoid_all()
        # set the test-electromagnets current to 5.5 A
        test.set_solenoid_current(5.5)
        # get the result that is being verified
        result = test.get_solenoid_B_flux(np.array([0, 0, 0]))

        # check result with predicted result
        prediction = np.array([np.sqrt(2) / 2 * 1.059787e-03, 0, np.sqrt(2) / 2 * 1.059787e-03])
        assert_array_almost_equal(result, prediction)