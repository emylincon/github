import unittest
import json
from api import app, VERSION


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """:arg
        this runs once at the start of test
        """
        cls.server = app.test_client()

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
        responses = [self.server.get(i)
                     for i in ('/latest', '/', f'/{VERSION}')]

        for response in responses:
            self.assertEqual(response.status_code, 200)

            result = json.loads(response.data.decode('utf-8'))
            assert type(result) is dict
            self.assertEqual(result['api_version'], VERSION)
            self.assertIsInstance(result, dict)
