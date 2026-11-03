import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from threading import Event

import app


async def collect(response):
    return [json.loads(item) async for item in response.body_iterator]


class ProgressTests(unittest.TestCase):
    def test_failed_index_save_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.json"
            index.touch()
            with patch.object(app, "SEARCH_INDEX_PATH", index), patch.object(app.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    app.save_search_index([])
            self.assertEqual(index.read_bytes(), b"")
            self.assertEqual(list(Path(directory).iterdir()), [index])

    def test_page_stages_include_ocr_translation_and_completion(self):
        events = []
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "")])
        with patch.object(app, "PdfReader", return_value=reader), patch.object(
            app, "ocr_page", return_value="ഭരണഭാഷ മാതൃഭാഷ"
        ), patch.object(app, "translate_malayalam_to_english", return_value="Official language"):
            passages = app.index_pdf(Path("scan.pdf"), {}, {}, events.append)
        self.assertEqual([event["stage"] for event in events], ["extracting", "ocr", "translating", "page_complete"])
        self.assertTrue(all(event["pages"] == 1 and event["page"] == 1 for event in events))
        self.assertEqual(passages[0].text, "Official language")

    def test_stream_delivers_warnings_and_terminal_result(self):
        def action(report):
            report({"stage": "extracting", "page": 1, "pages": 2})
            report({"stage": "warning", "message": "Blank page"})
            return {"passages": 3}
        response = app.indexing_response(action, True)
        events = asyncio.run(collect(response))
        self.assertEqual(events[-1], {"stage": "complete", "warnings": 1, "passages": 3})
        self.assertEqual(response.headers["x-accel-buffering"], "no")

    def test_failure_is_not_reported_as_success_and_lock_releases(self):
        def action(report):
            raise app.HTTPException(status_code=422, detail="No readable text")
        events = asyncio.run(collect(app.indexing_response(action, True)))
        self.assertEqual(events[-1], {"stage": "error", "message": "No readable text"})
        self.assertFalse(app.indexing_lock.locked())

    def test_simultaneous_indexing_is_rejected(self):
        started, release = Event(), Event()
        def action(report):
            started.set()
            release.wait(5)
            return {"passages": 1}
        response = app.indexing_response(action, True)
        try:
            self.assertTrue(started.wait(2))
            with self.assertRaises(app.HTTPException) as exc:
                app.indexing_response(action, True)
            self.assertEqual(exc.exception.status_code, 409)
        finally:
            release.set()
            asyncio.run(collect(response))

    def test_invalid_pdf_does_not_overwrite_existing_file(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "DATA_DIR", Path(directory)):
            path = Path(directory) / "existing.pdf"
            path.touch()
            upload = app.UploadFile(filename=path.name, file=BytesIO(b"%PDF-corrupt"))
            with self.assertRaises(app.HTTPException):
                app.upload_document(upload, progress=True)
            self.assertEqual(path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
