"""Unit tests for pycubeview.file_opening_utils module."""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from pycubeview.file_opening_utils import (
    open_txt_file,
    open_csv_file,
    open_hdr_file,
    open_wvl,
)


class TestOpenTxtFile(unittest.TestCase):
    """Test cases for open_txt_file function."""

    def test_open_txt_file_basic(self):
        """Test opening a basic txt file with comma-separated wavelengths."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("400.0,450.5,500.0,550.5,600.0")
            temp_path = Path(f.name)

        try:
            result = open_txt_file(temp_path)
            expected = np.array([400.0, 450.5, 500.0, 550.5, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_txt_file_single_value(self):
        """Test txt file with single wavelength value."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("550.0")
            temp_path = Path(f.name)

        try:
            result = open_txt_file(temp_path)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0], 550.0)
        finally:
            temp_path.unlink()

    def test_open_txt_file_with_trailing_space(self):
        """Test txt file with trailing space after last value."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("400.0,500.0,600.0 ")
            temp_path = Path(f.name)

        try:
            result = open_txt_file(temp_path)
            expected = np.array([400.0, 500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_txt_file_many_values(self):
        """Test txt file with many wavelength values."""
        wavelengths = [400.0 + i * 10 for i in range(100)]
        csv_str = ",".join(str(w) for w in wavelengths)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(csv_str)
            temp_path = Path(f.name)

        try:
            result = open_txt_file(temp_path)
            np.testing.assert_array_almost_equal(result, np.array(wavelengths))
        finally:
            temp_path.unlink()

    def test_open_txt_file_invalid_values(self):
        """Test txt file with non-numeric values raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("400.0,invalid,500.0")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                open_txt_file(temp_path)
        finally:
            temp_path.unlink()

    def test_open_txt_file_empty_file(self):
        """Test txt file that is empty."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            temp_path = Path(f.name)

        try:
            with self.assertRaises((ValueError, IndexError)):
                open_txt_file(temp_path)
        finally:
            temp_path.unlink()


class TestOpenCsvFile(unittest.TestCase):
    """Test cases for open_csv_file function."""

    def test_open_csv_file_basic(self):
        """Test opening a basic CSV file with wavelength column."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("wavelength,intensity\n")
            f.write("400.0,100\n")
            f.write("450.5,150\n")
            f.write("500.0,200\n")
            temp_path = Path(f.name)

        try:
            result = open_csv_file(temp_path)
            expected = np.array([400.0, 450.5, 500.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_csv_file_case_insensitive(self):
        """Test that wavelength column detection is case-insensitive."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("Wavelength,Value\n")
            f.write("500.0,10\n")
            f.write("600.0,20\n")
            temp_path = Path(f.name)

        try:
            result = open_csv_file(temp_path)
            expected = np.array([500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_csv_file_wavelength_not_first_column(self):
        """Test CSV where wavelength is not the first column."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("id,wavelength,intensity\n")
            f.write("1,400.0,100\n")
            f.write("2,500.0,200\n")
            temp_path = Path(f.name)

        try:
            result = open_csv_file(temp_path)
            expected = np.array([400.0, 500.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_csv_file_no_wavelength_column(self):
        """Test CSV file without wavelength column raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("value,intensity\n")
            f.write("100,50\n")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                open_csv_file(temp_path)
        finally:
            temp_path.unlink()

    def test_open_csv_file_single_row(self):
        """Test CSV file with only header and one data row."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("wavelength,intensity\n")
            f.write("550.0,100\n")
            temp_path = Path(f.name)

        try:
            result = open_csv_file(temp_path)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0], 550.0)
        finally:
            temp_path.unlink()

    def test_open_csv_file_many_columns(self):
        """Test CSV file with multiple columns."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("id,name,wavelength,intensity,quality\n")
            f.write("1,band1,400.0,100,good\n")
            f.write("2,band2,500.0,200,excellent\n")
            f.write("3,band3,600.0,150,good\n")
            temp_path = Path(f.name)

        try:
            result = open_csv_file(temp_path)
            expected = np.array([400.0, 500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()


class TestOpenHdrFile(unittest.TestCase):
    """Test cases for open_hdr_file function."""

    def test_open_hdr_file_basic(self):
        """Test opening a basic ENVI .hdr file with wavelength field."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("samples = 100\n")
            f.write("lines = 50\n")
            f.write("bands = 5\n")
            f.write("wavelength = {400.0, 450.0, 500.0, 550.0, 600.0}\n")
            temp_path = Path(f.name)

        try:
            result = open_hdr_file(temp_path)
            expected = np.array([400.0, 450.0, 500.0, 550.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_hdr_file_with_spaces(self):
        """Test .hdr file with varying spacing in wavelength field."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("wavelength = {400.0 , 450.0 , 500.0}\n")
            temp_path = Path(f.name)

        try:
            result = open_hdr_file(temp_path)
            expected = np.array([400.0, 450.0, 500.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_hdr_file_single_wavelength(self):
        """Test .hdr file with single wavelength value."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("wavelength = {550.0}\n")
            temp_path = Path(f.name)

        try:
            result = open_hdr_file(temp_path)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0], 550.0)
        finally:
            temp_path.unlink()

    def test_open_hdr_file_missing_wavelength_field(self):
        """Test .hdr file without wavelength field raises OSError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("samples = 100\n")
            f.write("lines = 50\n")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(OSError):
                open_hdr_file(temp_path)
        finally:
            temp_path.unlink()

    def test_open_hdr_file_many_wavelengths(self):
        """Test .hdr file with many wavelength values."""
        wavelengths = [400.0 + i * 5 for i in range(100)]
        wvl_str = ", ".join(str(w) for w in wavelengths)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write(f"wavelength = {{{wvl_str}}}\n")
            temp_path = Path(f.name)

        try:
            result = open_hdr_file(temp_path)
            np.testing.assert_array_almost_equal(result, np.array(wavelengths))
        finally:
            temp_path.unlink()

    def test_open_hdr_file_with_other_fields(self):
        """Test .hdr file with other fields interspersed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("description = {Test ENVI Header}\n")
            f.write("samples = 512\n")
            f.write("lines = 512\n")
            f.write("bands = 224\n")
            f.write("wavelength = {400.0, 401.79, 403.59, 405.39}\n")
            f.write("data type = 4\n")
            temp_path = Path(f.name)

        try:
            result = open_hdr_file(temp_path)
            expected = np.array([400.0, 401.79, 403.59, 405.39])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()


class TestOpenWvl(unittest.TestCase):
    """Test cases for open_wvl dispatcher function."""

    def test_open_wvl_txt_file(self):
        """Test opening .txt file through dispatcher."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("400.0,500.0,600.0")
            temp_path = Path(f.name)

        try:
            result = open_wvl(temp_path)
            expected = np.array([400.0, 500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_wvl_csv_file(self):
        """Test opening .csv file through dispatcher."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("wavelength,intensity\n")
            f.write("500.0,100\n")
            f.write("600.0,200\n")
            temp_path = Path(f.name)

        try:
            result = open_wvl(temp_path)
            expected = np.array([500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_wvl_hdr_file(self):
        """Test opening .hdr file through dispatcher."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hdr", delete=False
        ) as f:
            f.write("ENVI\n")
            f.write("wavelength = {400.0, 500.0, 600.0}\n")
            temp_path = Path(f.name)

        try:
            result = open_wvl(temp_path)
            expected = np.array([400.0, 500.0, 600.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()

    def test_open_wvl_with_string_path(self):
        """Test open_wvl accepts string path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("550.0")
            temp_path = f.name

        try:
            result = open_wvl(temp_path)
            self.assertAlmostEqual(result[0], 550.0)
        finally:
            Path(temp_path).unlink()

    def test_open_wvl_with_path_object(self):
        """Test open_wvl accepts Path object."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("550.0")
            temp_path = Path(f.name)

        try:
            result = open_wvl(temp_path)
            self.assertAlmostEqual(result[0], 550.0)
        finally:
            temp_path.unlink()

    def test_open_wvl_file_not_found(self):
        """Test open_wvl raises FileNotFoundError for nonexistent file."""
        nonexistent_path = Path("/nonexistent/path/file.txt")
        with self.assertRaises(FileNotFoundError):
            open_wvl(nonexistent_path)

    def test_open_wvl_unsupported_extension(self):
        """Test open_wvl raises ValueError for unsupported file extension."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("400.0")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                open_wvl(temp_path)
        finally:
            temp_path.unlink()

    def test_open_wvl_case_insensitive_extension(self):
        """Test open_wvl handles uppercase file extensions."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".TXT", delete=False
        ) as f:
            f.write("400.0,500.0")
            temp_path = Path(f.name)

        try:
            result = open_wvl(temp_path)
            expected = np.array([400.0, 500.0])
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
