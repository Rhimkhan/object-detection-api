import unittest
from utils.validators import allowed_file, validate_image_size
from io import BytesIO


class TestValidators(unittest.TestCase):
    def test_allowed_file_accepts_valid_extension(self):
        self.assertTrue(allowed_file("photo.jpg", {"jpg", "png"}))

    def test_allowed_file_rejects_invalid_extension(self):
        self.assertFalse(allowed_file("document.pdf", {"jpg", "png"}))

    def test_allowed_file_rejects_no_extension(self):
        self.assertFalse(allowed_file("noextension", {"jpg", "png"}))

    def test_validate_image_size_within_limit(self):
        fake_file = BytesIO(b"x" * 100)
        self.assertTrue(validate_image_size(fake_file, max_size_bytes=200))

    def test_validate_image_size_exceeds_limit(self):
        fake_file = BytesIO(b"x" * 300)
        self.assertFalse(validate_image_size(fake_file, max_size_bytes=200))


if __name__ == "__main__":
    unittest.main()
