import unittest

from numpy import int64
from Regression import BestModel
from github import Contributions, Transform


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


class TestContribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """:arg
        this runs once at the start of test
        """
        cls.con = Contributions()

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

    def test_contributions(self):
        data = self.con.get_query("emylincon")
        self.assertIsInstance(data, (list, dict))


class TestTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """:arg
        this runs once at the start of test
        """
        obj = Contributions()
        data = obj.get_query("emylincon")
        cls.trans = Transform(data)

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

    def test_most_contribution_day(self):
        data = self.trans.most_contribution_day()
        self.assertIsInstance(data.contribution, int64)


if __name__ == "__main__":
    unittest.main()
