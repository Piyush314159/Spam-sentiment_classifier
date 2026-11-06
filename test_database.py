import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.database import connect, load_corpus, save_evaluation


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "messages.tsv"
        self.source.write_text("ham\tHello!\nspam\tWin 'cash' now\nham\tHello!\n")

    def metrics(self):
        return {
            "evaluation": {"test_samples": 2, "selection_data": "validation only"},
            "naive_bayes": dict(threshold=.5, precision=1., recall=.5, f1=2/3, accuracy=.5, auc=.5),
        }

    def test_sql_preserves_order_duplicates_and_works_without_source(self):
        corpus = load_corpus(self.source)
        self.assertEqual(corpus.texts, ["Hello!", "Win 'cash' now", "Hello!"])
        self.assertEqual(corpus.labels, [0, 1, 0])
        self.assertEqual(load_corpus(self.source).dataset_id, corpus.dataset_id)
        self.source.unlink()
        self.assertEqual(load_corpus(self.source).texts, corpus.texts)

    def test_changed_source_creates_snapshot_without_removing_old_messages(self):
        first = load_corpus(self.source)
        self.source.write_text("spam\tNew snapshot\n")
        second = load_corpus(self.source)
        self.assertNotEqual(first.dataset_id, second.dataset_id)
        with connect(first.database_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM messages WHERE dataset_id=?", (first.dataset_id,)).fetchone()[0]
        self.assertEqual(count, 3)

    def test_prediction_links_and_error_query_survive_reopening(self):
        corpus = load_corpus(self.source)
        run = save_evaluation(corpus, self.metrics(), [2, 1], {"naive_bayes": ([.1, .2], [0, 0])})
        with connect(corpus.database_path) as conn:
            row = conn.execute("SELECT * FROM classification_errors WHERE run_id=?", (run,)).fetchone()
            self.assertEqual(row["position"], 1)
            self.assertEqual(row["text"], "Win 'cash' now")
            self.assertEqual(row["actual_label"], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 2)

    def test_invalid_prediction_rolls_back_whole_run(self):
        corpus = load_corpus(self.source)
        with self.assertRaises(sqlite3.IntegrityError):
            save_evaluation(corpus, self.metrics(), [2, 1], {"naive_bayes": ([.1, 2.], [0, 1])})
        with connect(corpus.database_path) as conn:
            for table in ("evaluation_runs", "model_metrics", "predictions"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_misaligned_outputs_rejected(self):
        corpus = load_corpus(self.source)
        with self.assertRaises(ValueError):
            save_evaluation(corpus, self.metrics(), [2, 1], {"naive_bayes": ([.1], [0])})


if __name__ == "__main__":
    unittest.main()
