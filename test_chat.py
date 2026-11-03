import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import app


class ChatTests(unittest.TestCase):
    def test_truncated_model_answer_is_not_returned(self):
        client = MagicMock()
        client.chat.completions.create.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="There are 1"))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", delta=SimpleNamespace(content=None))]),
        ]
        with patch.object(app, "get_client", return_value=(client, "test-model")):
            self.assertIsNone(app.generate_answer([]))

    def test_modules_document_beats_earlier_filename_matches(self):
        old = [app.Passage("Circular-KSMART.pdf", n, "KSMART GSPARK employee instructions") for n in range(1, 7)]
        modules = app.Passage("ksmart-modules-status-31.03.2024.pdf", 2, "K-SMART modules include Civil Registration and Property Tax.")
        with patch.object(app, "passages", old + [modules]):
            for question in ("ksmart modules?", "ksmart modules available now?"):
                self.assertEqual(app.retrieve(question)[0], modules)
            self.assertEqual(app.terms("K-SMART"), app.terms("ksmart"))

    def test_documents_are_newest_first_and_reuploads_move_to_top(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "DATA_DIR", Path(directory)):
            older = Path(directory) / "a.pdf"
            newer = Path(directory) / "z.pdf"
            older.touch()
            newer.touch()
            os.utime(older, ns=(100, 100))
            os.utime(newer, ns=(200, 200))
            self.assertEqual(app.list_documents()["documents"], ["z.pdf", "a.pdf"])
            os.utime(older, ns=(300, 300))
            self.assertEqual(app.list_documents()["documents"], ["a.pdf", "z.pdf"])

    def test_question_words_do_not_expand_from_history(self):
        history = [{"role": "assistant", "content": "Information systems support local government. Government staff process applications regularly keeping records."}]
        for question in ("What is GSPARK", "Gspark?"):
            self.assertEqual(app.resolve_acronyms(question, history), question)

    def test_only_explicit_matching_definitions_expand(self):
        for definition in ("Use the State Data Center (SDC).", "SDC (State Data Center)"):
            history = [{"role": "assistant", "content": definition}]
            self.assertEqual(app.resolve_acronyms("sdc cost", history), "sdc cost (State Data Center)")
        for text in ("State Data Center", "Unrelated Random Phrase (SDC)"):
            self.assertEqual(app.resolve_acronyms("SDC cost", [{"content": text}]), "SDC cost")

    def test_equivalent_questions_retrieve_same_evidence_with_history(self):
        history = [{"role": "assistant", "content": "Information systems process applications."}]
        passages = [
            app.Passage("Circular-KSMART.pdf", 1, "GSPARK stores employee details and IDs used in KSMART."),
            app.Passage("other.pdf", 1, "Information systems process applications."),
        ]
        with patch.object(app, "passages", passages):
            results = [app.retrieve(app.resolve_acronyms(q, history)) for q in ("What is GSPARK", "Gspark?")]
            self.assertEqual(results[0], [passages[0]])
            self.assertEqual(results[0], results[1])

    def test_no_evidence_claim_has_no_source_citations(self):
        with patch.object(app, "retrieve", return_value=[app.Passage("example.pdf", 1, "GSPARK employee details")]), patch.object(
            app, "generate_answer", return_value="The information is not available in the reference collection."
        ):
            answer, citations = app.answer_question("GSPARK launch budget?", [])
            self.assertTrue(answer)
            self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
