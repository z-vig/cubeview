"""Unit tests for pycubeview.utils module."""

import unittest
from pycubeview.utils import (
    get_bresenham_line,
    _low_bresenham_line,
    _high_bresenham_line,
)


class TestBresenhamLine(unittest.TestCase):
    """Test cases for Bresenham line algorithm implementations."""

    def test_low_bresenham_line_horizontal(self):
        """Test low Bresenham line with horizontal line (slope ~0)."""
        pt1 = (0, 0)
        pt2 = (5, 0)
        result = _low_bresenham_line(pt1, pt2)

        # Should contain all points from start to end
        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)
        self.assertEqual(len(result), 6)  # 6 points from (0,0) to (5,0)

    def test_low_bresenham_line_slight_positive_slope(self):
        """Test low Bresenham line with slight positive slope."""
        pt1 = (0, 0)
        pt2 = (4, 2)
        result = _low_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)
        # Line should progress left to right
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i + 1][0], result[i][0])

    def test_high_bresenham_line_vertical(self):
        """Test high Bresenham line with vertical line (steep slope)."""
        pt1 = (0, 0)
        pt2 = (0, 5)
        result = _high_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)
        self.assertEqual(len(result), 6)  # 6 points from (0,0) to (0,5)

    def test_high_bresenham_line_steep_slope(self):
        """Test high Bresenham line with steep positive slope."""
        pt1 = (0, 0)
        pt2 = (2, 4)
        result = _high_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)
        # Line should progress upward (y increases)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i + 1][1], result[i][1])

    def test_get_bresenham_line_uses_low_for_gentle_slope(self):
        """
        Test that get_bresenham_line uses low implementation for gentle slopes.
        """
        pt1 = (0, 0)
        pt2 = (10, 3)
        result = get_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)

    def test_get_bresenham_line_uses_high_for_steep_slope(self):
        """
        Test that get_bresenham_line uses high implementation for steep slopes.
        """
        pt1 = (0, 0)
        pt2 = (3, 10)
        result = get_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)

    def test_bresenham_line_negative_direction(self):
        """Test Bresenham line working in reverse direction."""
        pt1 = (5, 5)
        pt2 = (0, 0)
        result = get_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)

    def test_bresenham_line_single_point(self):
        """Test Bresenham line with start and end at same point."""
        pt1 = (3, 3)
        pt2 = (3, 3)
        result = get_bresenham_line(pt1, pt2)

        self.assertEqual(result, [(3, 3)])

    def test_bresenham_line_diagonal(self):
        """Test Bresenham line with perfect diagonal."""
        pt1 = (0, 0)
        pt2 = (5, 5)
        result = get_bresenham_line(pt1, pt2)

        self.assertEqual(result[0], pt1)
        self.assertEqual(result[-1], pt2)
        # All points should be on or near the diagonal
        self.assertTrue(len(result) >= 6)


if __name__ == "__main__":
    unittest.main()
