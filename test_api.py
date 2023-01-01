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
        self.test_username = "emylincon"

    def tearDown(self):
        """:arg
        this runs after each test
        """
        pass

    def test_root(self):
        responses = [self.server.get(i)
                     for i in ('/latest', '/', f'/{VERSION}')]

        for response in responses:
            self.assertEqual(response.status_code, 200)

            result = json.loads(response.data.decode('utf-8'))
            self.assertEqual(result['api_version'], VERSION)
            self.assertIsInstance(result, dict)

    def test_contributions(self):
        """
        test endpoint: "/latest/<string:username>/contributions/day/<string:kind>
        """
        def get_url(version: str, kind: str) -> str:
            return f"/{version}/{self.test_username}/contributions/day/{kind}"
        kinds = ("average", "least", "most")
        urls = [get_url("latest", i) for i in kinds] + \
            [get_url(VERSION, i) for i in kinds]
        responses = [self.server.get(i)
                     for i in urls]

        for response in responses:
            self.assertEqual(response.status_code, 200)

            result = json.loads(response.data.decode('utf-8'))
            self.assertEqual(result['api_version'], VERSION)
            self.assertIsInstance(result, dict)
