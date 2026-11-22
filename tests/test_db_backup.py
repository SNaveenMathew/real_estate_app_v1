"""Backups, WAL recovery accounting, and the checkpoint that close() performs."""

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import db_backup  # noqa: E402  (scripts/ is not an importable package)


def _make_db(path: Path, rows: int = 1) -> None:
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE houses (house_id VARCHAR, price DOUBLE)")
    for i in range(rows):
        conn.execute("INSERT INTO houses VALUES (?, ?)", [f"h{i}", 100.0 + i])
    conn.execute("CHECKPOINT")
    conn.close()


def _house_count(path: Path) -> int:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM houses").fetchone()[0])
    finally:
        conn.close()


def test_backup_then_restore_round_trips(tmp_path):
    db = tmp_path / "real_estate.duckdb"
    _make_db(db, rows=1)

    dest = db_backup.backup(db, tmp_path / "backups", label="test")
    assert dest.exists()
    assert dest.name.endswith("-test.duckdb")
    assert not dest.with_suffix(".duckdb.part").exists()  # only renamed after verify
    info = db_backup.verify(dest)
    assert info["rows"]["houses"] == 1

    conn = duckdb.connect(str(db))
    conn.execute("INSERT INTO houses VALUES ('h2', 200.0)")
    conn.execute("CHECKPOINT")
    conn.close()
    assert _house_count(db) == 2

    db_backup.restore(dest, db)
    assert _house_count(db) == 1
    assert list(tmp_path.glob("real_estate.pre-restore-*.duckdb"))  # the replaced file is kept


def test_restore_refuses_something_that_is_not_a_database(tmp_path):
    db = tmp_path / "real_estate.duckdb"
    _make_db(db)
    bad = tmp_path / "bad.duckdb"
    bad.write_bytes(b"this is not a duckdb file")

    with pytest.raises(Exception):
        db_backup.restore(bad, db)
    assert _house_count(db) == 1  # untouched


def test_backup_refuses_a_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        db_backup.backup(tmp_path / "nope.duckdb", tmp_path / "backups")


def test_a_quarantined_wal_is_counted_and_moved_aside(tmp_path, monkeypatch):
    import db.duckdb_store as store

    db = tmp_path / "db.duckdb"
    _make_db(db)
    (tmp_path / "db.duckdb.wal").write_bytes(b"not a real wal")

    real_connect = duckdb.connect
    calls = {"n": 0}

    def flaky(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise duckdb.IOException("replaying WAL file: corrupt entry")
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky)
    before = store.wal_recovery_count()
    conn = store._safe_connect(db)
    try:
        assert store.wal_recovery_count() == before + 1
        assert list(tmp_path.glob("db.duckdb.wal.bad.*"))
    finally:
        conn.close()


def test_close_checkpoints_so_the_data_survives(reference_data):
    """close() runs CHECKPOINT before disconnecting; the rows must be on disk afterwards."""
    import db.duckdb_store as store
    from config import settings

    db_path = Path(settings.duckdb_path)
    store.close()
    assert db_path.exists()
    assert _house_count(db_path) >= 1
