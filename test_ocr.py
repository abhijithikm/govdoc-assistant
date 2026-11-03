import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app


class OCRTests(unittest.TestCase):
    def test_repetitive_transcription_is_rejected(self):
        self.assertFalse(app.usable_ocr("അപേക്ഷിക്കുക ഉത്തരവാദിത്തം " * 30))
        self.assertTrue(app.usable_ocr("ഭരണഭാഷ - മാതൃഭാഷ"))

    def test_english_only_installation_falls_back(self):
        with patch.object(app, "tesseract_command", return_value=("tesseract", {})), patch.object(
            app.subprocess, "run", return_value=SimpleNamespace(stdout="eng\nosd\n")
        ) as run:
            self.assertIsNone(app.tesseract_ocr(Path("scan.png")))
            self.assertEqual(run.call_count, 1)

    def test_legacy_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "scan.pdf"
            pdf.touch()
            stat = pdf.stat()
            legacy_key = f"scan.pdf:{stat.st_mtime_ns}:{stat.st_size}:page:1"
            cache = {legacy_key: "incorrect old transcription"}
            with patch.object(app, "render_pdf_page", return_value=Path("scan.png")), patch.object(
                app, "tesseract_ocr", return_value="ഭരണഭാഷ - മാതൃഭാഷ"
            ) as local_ocr, patch.object(app, "vision_ocr") as vision:
                self.assertEqual(app.ocr_page(pdf, 1, cache), "ഭരണഭാഷ - മാതൃഭാഷ")
                self.assertEqual(app.ocr_page(pdf, 1, cache), "ഭരണഭാഷ - മാതൃഭാഷ")
                local_ocr.assert_called_once()
                vision.assert_not_called()


if __name__ == "__main__":
    unittest.main()
