import unittest
from Regression import BestModel


class TestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """:arg
        this runs once at the start of test
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """:arg
        this runs once after all test is completed
        """
        print("teardownClass")

    def setUp(self):
        """:arg
        this runs before each test
        """
        pass

    def tearDown(self):
        """:arg
        this runs after each test
        """
        pass

    def test_regression(self):
        xi = [5, 15, 25, 35, 45, 55]
        yi = [5, 20, 14, 32, 22, 38]
        my_obj = BestModel(xi, yi)
        p = my_obj.compute_best_model()
        result = p.predict([25])[0]
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
