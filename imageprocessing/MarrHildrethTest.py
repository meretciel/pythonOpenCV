import unittest
import numpy as np

from imageprocessing.MarrHildreth import zeroCrossing


class TestMarrHildreth(unittest.TestCase):

    def test_1(self):
        image = np.asarray(
            [[1,1,1],
            [1,1,1],
            [1,1,1]]
        )

        expected = np.full((3,3), False)

        self.assertTrue(np.all(zeroCrossing(image) == expected))

    def test_2(self):
        image = np.asarray([
            [5,5,1],
            [5,1,1],
            [1,-5,-5]
        ])

        expected = np.full((3,3), False)
        expected[1,1] = True

        self.assertTrue(np.all(zeroCrossing(image) == expected))

if __name__ == '__main__':
    unittest.main()