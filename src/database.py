"""SQLite corpus storage and reproducible evaluation/prediction history."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "spam.sqlite3"


@contextmanager
def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize(path):
    with connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY, source_path TEXT NOT NULL,
                sha256 TEXT NOT NULL, imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_path, sha256)
            );
            CREATE TABLE IF NOT EXISTS messages (
                dataset_id INTEGER NOT NULL REFERENCES datasets(id),
                position INTEGER NOT NULL, text TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0, 1)),
                PRIMARY KEY(dataset_id, position)
            );
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
                protocol_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(id, dataset_id)
            );
            CREATE TABLE IF NOT EXISTS model_metrics (
                run_id INTEGER NOT NULL REFERENCES evaluation_runs(id), model TEXT NOT NULL,
                threshold REAL NOT NULL, precision REAL NOT NULL, recall REAL NOT NULL,
                f1 REAL NOT NULL, accuracy REAL NOT NULL, auc REAL NOT NULL,
                final_train_loss REAL, PRIMARY KEY(run_id, model)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                run_id INTEGER NOT NULL, dataset_id INTEGER NOT NULL,
                model TEXT NOT NULL, position INTEGER NOT NULL,
                probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
                predicted_label INTEGER NOT NULL CHECK(predicted_label IN (0, 1)),
                PRIMARY KEY(run_id, model, position),
                FOREIGN KEY(run_id, model) REFERENCES model_metrics(run_id, model),
                FOREIGN KEY(run_id, dataset_id) REFERENCES evaluation_runs(id, dataset_id),
                FOREIGN KEY(dataset_id, position) REFERENCES messages(dataset_id, position)
            );
            CREATE VIEW IF NOT EXISTS classification_errors AS
                SELECT p.*, m.text, m.label AS actual_label
                FROM predictions p JOIN messages m USING(dataset_id, position)
                WHERE p.predicted_label != m.label;
        """)


@dataclass
class Corpus:
    database_path: Path
    dataset_id: int
    texts: list
    labels: list


def load_corpus(filepath, database_path=None):
    """Import changed TSV snapshots; always return messages in SQL row order.

    Duplicate texts retain distinct positions so the existing seeded splits
    remain identical. If the original file is absent, use its latest snapshot.
    """
    source = Path(filepath).resolve()
    path = Path(database_path) if database_path is not None else source.parent / "spam.sqlite3"
    initialize(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
    with connect(path) as conn:
        if digest is None:
            row = conn.execute(
                "SELECT id FROM datasets WHERE source_path=? ORDER BY id DESC LIMIT 1", (str(source),)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM datasets WHERE source_path=? AND sha256=?", (str(source), digest)
            ).fetchone()
        if row is None:
            if digest is None:
                raise FileNotFoundError(f"No source file or SQL snapshot for {source}")
            frame = pd.read_csv(source, sep="\t", header=None, usecols=[0, 1], names=["label", "message"])
            labels = frame.label.map({"ham": 0, "spam": 1})
            if labels.isna().any() or frame.message.isna().any():
                raise ValueError("Messages must have a ham/spam label and non-empty text")
            dataset_id = conn.execute(
                "INSERT INTO datasets(source_path, sha256) VALUES (?, ?)", (str(source), digest)
            ).lastrowid
            conn.executemany(
                "INSERT INTO messages(dataset_id, position, text, label) VALUES (?, ?, ?, ?)",
                ((dataset_id, i, text, int(label)) for i, (text, label) in
                 enumerate(zip(frame.message, labels))),
            )
        else:
            dataset_id = row["id"]
        rows = conn.execute(
            "SELECT text, label FROM messages WHERE dataset_id=? ORDER BY position", (dataset_id,)
        ).fetchall()
    return Corpus(path, dataset_id, [r["text"] for r in rows], [r["label"] for r in rows])


def save_evaluation(corpus, metrics, test_positions, model_outputs):
    """Commit metrics and every held-out prediction together, linked to messages."""
    positions = [int(i) for i in test_positions]
    if len(set(positions)) != len(positions) or any(i < 0 or i >= len(corpus.texts) for i in positions):
        raise ValueError("Test positions must be unique valid corpus positions")
    if len(positions) != metrics["evaluation"]["test_samples"]:
        raise ValueError("Prediction count does not match evaluation split")
    if set(model_outputs) != set(metrics) - {"evaluation"}:
        raise ValueError("Every evaluated model must have predictions")
    for scores, labels in model_outputs.values():
        if len(scores) != len(positions) or len(labels) != len(positions):
            raise ValueError("Scores and labels must match test positions")
    with connect(corpus.database_path) as conn:
        run_id = conn.execute(
            "INSERT INTO evaluation_runs(dataset_id, protocol_json) VALUES (?, ?)",
            (corpus.dataset_id, json.dumps(metrics["evaluation"])),
        ).lastrowid
        for model, (scores, labels) in model_outputs.items():
            m = metrics[model]
            conn.execute(
                "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, model, *[float(m[k]) for k in
                 ("threshold", "precision", "recall", "f1", "accuracy", "auc")],
                 float(m["final_train_loss"]) if "final_train_loss" in m else None),
            )
            conn.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                ((run_id, corpus.dataset_id, model, pos, float(score), int(label))
                 for pos, score, label in zip(positions, scores, labels)),
            )
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Query saved classifier metrics or misclassified messages.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression"], default="naive_bayes")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be 1–200")
    initialize(args.database)
    with connect(args.database) as conn:
        if args.errors:
            rows = conn.execute(
                "SELECT position, probability, predicted_label, actual_label, text "
                "FROM classification_errors WHERE run_id=(SELECT MAX(id) FROM evaluation_runs) "
                "AND model=? ORDER BY position LIMIT ?", (args.model, args.limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM model_metrics WHERE run_id=(SELECT MAX(id) FROM evaluation_runs) ORDER BY model"
            ).fetchall()
    print(json.dumps([dict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
