import unittest
import numpy as np
from src.main.python.Electromagnet import Electromagnet

class TestElectromagnet(unittest.TestCase):

    def setUp(self):
        pass

    def test_d_l(self):
        test = Electromagnet(0.01, ['x', np.pi / 6], 350, 0.005, 0.01)
        self.assertEqual(test.get_d_c, 350 * 0.005 ** 2 / )

