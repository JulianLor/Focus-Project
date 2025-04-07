import unittest
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from src.main.python.PermanentMagnet import PermanentMagnet

class TestElectromagnet(unittest.TestCase):

    def setUp(self):
        pass

    # testing whether the returned volume is correct
    def test_get_volume_FEM(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.1,0.1,0.1]), ['y', 0])
        # define the FEM cube_size
        test.cube_size = 0.1
        # get the result that is being verified
        result = test.get_volume_FEM()

        # check result with predicted result
        prediction = 0.001
        self.assertAlmostEqual(result, prediction)

    # testing if the system exit is triggered for get_solenoid_B_flux function
    def test_get_volume_FEM_2(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.1,0.1,0.1]), ['y', 0])

        # check if system exit is triggered
        result = test.get_volume_FEM()

        # Check exit code
        prediction = 0.01 * 0.01 * 0.01
        self.assertAlmostEqual(result, prediction)

    # testing whether the returned moment vector is correct
    def test_get_moment_vector(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.1,0.1,0.1]), ['y', np.pi/2])
        # get the result that is being verified
        result = test.get_moment_vector()

        # check result with predicted result
        prediction = np.array([1000,0,0])
        assert_array_almost_equal(result, prediction)

    # testing whether the returned moment vector is correct
    def test_get_FEM_number(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.01,0.05,0.1]), ['y', np.pi/2])
        # define the FEM cube_size
        test.cube_size = 0.01
        # get the result that is being verified
        result = test.get_FEM_number()

        # check result with predicted result
        prediction = 50
        self.assertEqual(result, prediction)

    # testing whether the returned moment vector is correct
    def test_get_FEM_number_2(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.01,0.05,0.1]), ['y', np.pi/2])
        # define the FEM cube_size
        test.cube_size = 0.03

        # check if system exit is triggered
        with self.assertRaises(SystemExit) as cm:
            test.get_FEM_number()  # Should trigger sys.exit()

        # Check exit code
        self.assertEqual(cm.exception.code, 'FEM dimensions do not work out')

    # testing whether the cube_centers are returned correctly
    def test_divide_magnet(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.2, 0.2, 0.1]), ['y', 0])
        # define the FEM cube_size
        test.cube_size = 0.1
        # get the result that is being verified
        result = test.divide_magnet()

        # check result with predicted result
        prediction = np.array([[-0.05, -0.05, 0.],
                               [0.05, -0.05, 0.],
                               [-0.05, 0.05, 0.],
                               [0.05, 0.05, 0.]])
        assert_array_almost_equal(result, prediction)

    # testing whether the magnetisation vector is returned correctly (for FEM)
    def test_get_magnetisation(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.1, 0.1, 0.1]), ['x', np.pi / 2])
        # define the FEM cube_size
        test.cube_size = 0.01
        # get the result that is being verified
        result = test.get_magnetisation('FEM')

        # check result with predicted result
        prediction = np.array([0,-0.001,0])
        assert_array_almost_equal(result, prediction)

    # testing whether the magnetisation vector is returned correctly (for magnet)
    def test_get_magnetisation_2(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.1, 0.1, 0.1]), ['x', np.pi / 2])
        # get the result that is being verified
        result = test.get_magnetisation('Magnet')

        # check result with predicted result
        prediction = np.array([0,-1,0])
        assert_array_almost_equal(result, prediction)

    # testing whether a non-valid mode is recognised as such
    def test_get_magnetisation_3(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.01,0.05,0.1]), ['y', np.pi/2])

        # check if system exit is triggered
        with self.assertRaises(SystemExit) as cm:
            test.get_magnetisation('test')  # Should trigger sys.exit()

        # Check exit code
        self.assertEqual(cm.exception.code, 'Invalid mode')

    # testing whether if system exit is triggered with missing cube size
    def test_get_magnetisation_4(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.01,0.05,0.1]), ['x', np.pi/2])

        # get the result that is being verified
        result = test.get_magnetisation('FEM')

        # check result with predicted result
        prediction = np.array([0, -0.001, 0])
        assert_array_almost_equal(result, prediction)

    # testing whether the flux for a FEM magnet is returned correctly
    def test_get_B_flux(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.001, 0.001, 0.001]), ['y', 0])
        # define the FEM cube_size
        test.cube_size = 0.001
        # get the result that is being verified
        result = test.get_B_flux(np.array([0.01,0,0]))

        # check result with predicted result
        prediction = np.array([0, 0, -1*10**(-7)])
        assert_array_almost_equal(result, prediction)

    # testing if the flux for the entire magnet (FEM method) is returned correctly
    def test_get_magnet_B_flux(self):
        # setup test instance
        test = PermanentMagnet([0, 0, 0], 1000, np.array([0.001, 0.002, 0.001]), ['y', 0])
        # define the FEM cube_size
        test.cube_size = 0.001
        # get the result that is being verified
        result = test.get_magnet_B_flux(np.array([0.01, 0, 0]))

        # check result with predicted result
        prediction = test.get_B_flux(np.array([0.01, -0.0005, 0])) * 2
        assert_array_almost_equal(result, prediction)

