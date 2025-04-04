import unittest
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from src.main.python.Electromagnet import Electromagnet
from src.main.python.PermanentMagnet import PermanentMagnet
from src.main.python.ActuationSystem import ActuationSystem

class TestActuationSystem(unittest.TestCase):

    def setUp(self):
        pass

    # testing whether the system parameters are correct
    def test_check_system_param(self):
        # setup test instance
        pos_ElectroMagnet = [0.1, 'y', np.pi / 4], [0.1, 'x', -np.pi / 4], [0.1, 'y', -np.pi / 4], [0.1, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        result = test.check_system_param()

        # check result with predicted result
        prediction = True
        self.assertEqual(result, prediction)

    # check if the electromagnet parameters are set correctly
    def test_set_Electromagnet_param(self):
        # setup test instance
        pos_ElectroMagnet = [0.13, 'y', np.pi / 4], [0.13, 'x', -np.pi / 4], [0.13, 'y', -np.pi / 4], [0.13, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        test.set_Electromagnet_param(200, 0.05, 0.03)
        result = test.param_ElectroMagnet

        # check result with predicted result
        prediction = np.array([200, 0.05, 0.03])
        assert_array_almost_equal(result, prediction)

    # check if the electromagnet parameters are set correctly with empty entries
    def test_set_Electromagnet_param_2(self):
        # setup test instance
        pos_ElectroMagnet = [0.125, 'y', np.pi / 4], [0.125, 'x', -np.pi / 4], [0.125, 'y', -np.pi / 4], [0.125, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        test.set_Electromagnet_param(0, 0.05, 0.03)
        result = test.param_ElectroMagnet

        # check result with predicted result
        prediction = np.array([400, 0.05, 0.03])
        assert_array_almost_equal(result, prediction)

    # check if the electromagnets parameters are returned in the correct form
    def test_get_Electromagnet_param(self):
        # setup test instance
        pos_ElectroMagnet = [0.12, 'y', np.pi / 4], [0.12, 'x', -np.pi / 4], [0.12, 'y', -np.pi / 4], [0.12, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        test.set_Electromagnet_param(0, 0.05, 0.03)
        result = test.get_Electromagnet_param()

        # check result with predicted result
        d_z = [0.12, 0.12, 0.12, 0.12]
        angle = [['y', np.pi / 4], ['x', -np.pi / 4], ['y', -np.pi / 4], ['x', np.pi / 4]]
        prediction = d_z, angle, 400, 0.05, 0.03
        self.assertTupleEqual(result, prediction)

    # check if the electromagnets are generated correctly
    def test_generate_Electromagnet(self):
        # setup test instance
        pos_ElectroMagnet = [0.11, 'y', np.pi / 4], [0.11, 'x', -np.pi / 4], [0.11, 'y', -np.pi / 4], [0.11, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        test.set_Electromagnet_param(200, 0.05, 0.03)
        test.generate_Electromagnets()
        result = test.Electromagnets

        # generate expected result
        angle_Electromagnets = ['y', np.pi / 4], ['x', -np.pi / 4], ['y', -np.pi / 4], ['x', np.pi / 4]

        # do checks if output corresponds to expected result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        for i, instance in enumerate(result):
            self.assertIsInstance(instance, Electromagnet)
            self.assertAlmostEqual(instance.d_z, 0.11)
            self.assertListEqual(instance.angle, angle_Electromagnets[i])
            self.assertEqual(instance.n, 200)
            self.assertAlmostEqual(instance.r_in, 0.05)
            self.assertAlmostEqual(instance.l_c, 0.03)

    # check if the Permanent Magnet parameters are set correctly
    def test_set_PermMagnet_param(self):
        # setup test instance
        pos_ElectroMagnet = [0.3, 'y', np.pi / 4], [0.3, 'x', -np.pi / 4], [0.3, 'y', -np.pi / 4], [0.3, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        angle = (['y', 0], ['y', 0], ['y', 0], ['y', 0])
        test.set_PermMagnet_param(tuple(angle), 1000, np.array([0.7,0.53,0.51]))
        result = test.param_PermMagnet

        # check result with predicted result
        prediction = angle, 1000, np.array([0.7,0.53,0.51])
        self.assertTupleEqual((result[0], result[1]), (prediction[0], prediction[1]))
        assert_array_almost_equal(result[2], prediction[2])

    # check if the Permanent Magnet parameters are set correctly
    def test_set_PermMagnet_param_2(self):
        # setup test instance
        pos_ElectroMagnet = [0.4, 'y', np.pi / 4], [0.4, 'x', -np.pi / 4], [0.4, 'y', -np.pi / 4], [0.4, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.85, 0, 0], [0, 0.85, 0], [-0.85, 0, 0], [0, -0.85, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        angle = (['y', 0], ['y', 0], ['y', 0], ['y', 0])
        test.set_PermMagnet_param(tuple(angle), 0, np.array([0.7, 0.53, 0.51]))
        result = test.param_PermMagnet

        # check result with predicted result
        prediction = angle, 1153873.3374162412, np.array([0.07, 0.03, 0.01])
        self.assertTupleEqual((result[0], result[1]), (prediction[0], prediction[1]))
        assert_array_almost_equal(result[2], prediction[2])

    # check if the parameters are returned in an appropriate form
    def test_get_PermMagnet_param(self):
        # setup test instance
        pos_ElectroMagnet = [0.12, 'y', np.pi / 4], [0.12, 'x', -np.pi / 4], [0.12, 'y', -np.pi / 4], [0.12, 'x', np.pi / 4]
        pos_PermMagnet = np.array([[0.285, 0, 0], [0, 0.285, 0], [-0.285, 0, 0], [0, -0.285, 0]])

        test = ActuationSystem(4, 4, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        angle = [['y', 0], ['y', 0], ['y', 0], ['y', 0]]
        test.set_PermMagnet_param(tuple(angle), 1000, np.array([0.7, 0.53, 0.51]))
        result = test.get_PermMagnet_param()

        # check result with predicted result
        prediction = pos_PermMagnet, 1000, np.array([0.7, 0.53, 0.51]), angle
        assert_array_almost_equal(result[0], prediction[0])
        assert_array_almost_equal(result[2], prediction[2])
        self.assertTupleEqual((result[1], result[3]), (prediction[1], prediction[3]))

    # check if the permanent magnets are generated correctly
    def test_generate_PermMagnets(self):
        # setup test instance
        pos_ElectroMagnet = [0.11, 'y', np.pi / 4], [0.11, 'x', -np.pi / 4], [0.11, 'y', -np.pi / 4], [0.11, 'x', np.pi / 4]
        a = 0.285
        b = np.sqrt(2) * 0.285 / 2
        c = 0.105
        pos_PermMagnet = np.array([[a, 0, c], [b, b, c], [0, a, 0.105], [-b, b, 0.105],
                                   [-a, -0, 0.105], [-b,-b,0.105], [0, -a, 0.105], [b, -b, c]])

        test = ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        angle = [['y', 0], ['y', 0], ['y', 0], ['y', 0],
                 ['y', 0], ['y', 0], ['y', 0], ['y', 0]]
        test.set_PermMagnet_param(tuple(angle), 2000, np.array([0.7, 0.03, 0.01]))
        test.generate_PermMagnets()
        result = test.PermMagnets

        # do checks if output corresponds to expected result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 8)
        for i, instance in enumerate(result):
            self.assertIsInstance(instance, PermanentMagnet)
            assert_array_almost_equal(instance.pos, pos_PermMagnet[i])
            self.assertAlmostEqual(instance.magnet_moment, 2000)
            assert_array_almost_equal(instance.magnet_dim, np.array([0.7, 0.03, 0.01]))
            self.assertListEqual(instance.angle, angle[i])

    # check if the flux output of the cancellation system is correct
    def test_get_canc_flux(self):
        # setup test instance
        pos_ElectroMagnet = [0.11, 'y', np.pi / 4], [0.11, 'x', -np.pi / 4], [0.11, 'y', -np.pi / 4], [0.11, 'x',  np.pi / 4]
        a = 0.285
        b = np.sqrt(2) * 0.285 / 2
        c = 0.105
        pos_PermMagnet = np.array([[a, 0, c], [b, b, c], [0, a, 0.105], [-b, b, 0.105],
                                   [-a, -0, 0.105], [-b, -b, 0.105], [0, -a, 0.105], [b, -b, c]])

        test = ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        angle = [['y', 0], ['y', 0], ['y', 0], ['y', 0],
                 ['y', 0], ['y', 0], ['y', 0], ['y', 0]]
        test.set_PermMagnet_param(tuple(angle), 10000, np.array([0.7, 0.03, 0.01]))
        test.generate_PermMagnets()
        result = test.get_canc_flux(np.array([0,0,0]))

        # check result with predicted result
        prediction = np.array([0,0,0])
        for i in range(8):
            prediction_magnet = PermanentMagnet(pos_PermMagnet[i], 10000, np.array([0.7, 0.03, 0.01]), angle[i])
            prediction = np.add(prediction, prediction_magnet.get_magnet_B_flux(np.array([0, 0, 0]))) # r vector to point of interest
        assert_array_almost_equal(result, prediction)

    # check if the flux output of the RMF system is correct
    def test_get_RMF_flux(self):
        # setup test instance
        pos_ElectroMagnet = [0.11, 'y', np.pi / 4], [0.11, 'x', -np.pi / 4], [0.11, 'y', -np.pi / 4], [0.11, 'x',
                                                                                                       np.pi / 4]
        a = 0.285
        b = np.sqrt(2) * 0.285 / 2
        c = 0.105
        pos_PermMagnet = np.array([[a, 0, c], [b, b, c], [0, a, 0.105], [-b, b, 0.105],
                                   [-a, -0, 0.105], [-b, -b, 0.105], [0, -a, 0.105], [b, -b, c]])

        test = ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnet)

        # get the result that is being verified
        test.set_Electromagnet_param(400, 0.05, 0.03)
        test.generate_Electromagnets()
        result = test.get_RMF_flux(np.array([1,1,1,1]), np.array([0,0,0]))

        # generate expected result
        angle_Electromagnets = ['y', np.pi / 4], ['x', -np.pi / 4], ['y', -np.pi / 4], ['x', np.pi / 4]

        # check result with predicted result
        prediction = np.array([0, 0, 0])
        for i in range(4):
            prediction_magnet = Electromagnet(0.11, angle_Electromagnets[i], 400, 0.05, 0.03)
            prediction_magnet.set_solenoid_current(1.0)
            prediction_magnet.save_solenoid_all()
            prediction = np.add(prediction, prediction_magnet.get_solenoid_B_flux(np.array([0, 0, 0])))
        assert_array_almost_equal(result, prediction)

