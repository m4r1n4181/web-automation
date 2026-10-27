import pytest
import sqlite3
from src.database import Database


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_payments.db"
    return Database(db_file)


def test_aliases(db):
    # Save alias
    db.save_alias("PDF NAME", "WEB NAME", source="manual")

    # Retrieve
    alias = db.get_alias("PDF NAME")
    assert alias is not None
    assert alias['web_name'] == "WEB NAME"
    assert alias['used_count'] == 1

    # Unknown returns None
    assert db.get_alias("UNKNOWN") is None

    # Upsert updates
    db.save_alias("PDF NAME", "NEW WEB NAME", source="manual")
    alias = db.get_alias("PDF NAME")
    assert alias['web_name'] == "NEW WEB NAME"
    assert alias['used_count'] == 2


def test_processing_log(db):
    run_id = db.start_run("test.pdf", "Coordinates")
    assert run_id > 0

    db.finish_run(run_id, total=10, matched=8, manual=1, unresolved=1)

    stats = db.get_stats()
    assert stats['total_runs'] == 1
    assert stats['total_processed'] == 10
    assert stats['total_matched'] == 8
    assert stats['accuracy_pct'] == 80.0


def test_unresolved(db):
    db.add_unresolved("UNRESOLVED NAME", notes="Not found")

    unresolved = db.get_unresolved()
    assert len(unresolved) == 1
    assert unresolved[0]['pdf_name'] == "UNRESOLVED NAME"
    assert unresolved[0]['seen_count'] == 1

    # Increment seen count
    db.add_unresolved("UNRESOLVED NAME", notes="Still not found")
    unresolved = db.get_unresolved()
    assert unresolved[0]['seen_count'] == 2


def test_get_stats_empty(db):
    stats = db.get_stats()
    assert stats['total_aliases'] == 0
    assert stats['total_runs'] == 0
    assert stats['accuracy_pct'] == 0
