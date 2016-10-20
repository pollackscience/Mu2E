import unittest

class FooTests(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

    @unittest.skip("skipping autofail")
    def test_false(self):
        self.assertFalse(True)

    @unittest.skip("skipping autofail")
    def test_key(self):
        a = ['a','b']
        b = ['b']
        self.assertEqual(a,b)
