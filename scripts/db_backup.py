#!/usr/bin/env python3
"""Back up, verify and restore the DuckDB database.

Everything the app knows - houses, the catalog, uploaded datasets, cache rows - is one
file, and until now the only copies of it were the ones the operating system happened to
make. Nine ``*.wal.bad.*`` files in ``data/`` are evidence that the recovery path runs in
practice; nothing ever took a snapshot first.

The three commands are deliberately small and independent of the running app, so they work
when the app does not:

    python scripts/db_backup.py backup                    # checkpoint, copy, verify
    python scripts/db_backup.py list                      # what is in data/backups/
    python scripts/db_backup.py verify FILE               # open read-only, count tables
    python scripts/db_backup.py restore FILE [--force]    # stop the app first

``backup`` checkpoints first (a .wal that is not flushed would not be in the copy), writes
to a ``.part`` file and renames it only after ``verify`` succeeds, so an interrupted backup
can never look like a good one. ``restore`` refuses to run while another process holds the
database - that refusal is the feature, because copying over a live DuckDB file is how a
database gets corrupted. It also keeps the current file as ``<name>.pre-restore-<ts>``
before replacing it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
import db.duckdb_store as store

DEFAULT_BACKUP_DIR = settings.data_dir / "backups"


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def verify(path: Path) -> dict:
    """Open the file read-only and report what is in it. Raises if it cannot be opened."""
    import duckdb

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    conn = duckdb.connect(str(path), read_only=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()]
        counts = {}
        for name in ("houses", "sold_homes", "nri_tracts"):
            if name in tables:
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
    finally:
        conn.close()
    return {"path": str(path), "bytes": path.stat().st_size, "tables": len(tables), "rows": counts}


def backup(db_path: Path | None = None, out_dir: Path | None = None, label: str = "") -> Path:
    """Checkpoint, copy and verify. Returns the backup path."""
    db_path = Path(db_path or settings.duckdb_path)
    out_dir = Path(out_dir or DEFAULT_BACKUP_DIR)
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        store.checkpoint(db_path)
    except Exception as exc:                    # the app is running and holds the lock
        print(f"warning: could not CHECKPOINT ({exc}); copying the file as it stands", file=sys.stderr)

    suffix = f"-{label}" if label else ""
    dest = out_dir / f"{db_path.stem}-{_stamp()}{suffix}.duckdb"
    part = dest.with_suffix(".duckdb.part")
    shutil.copy2(db_path, part)
    wal = Path(str(db_path) + ".wal")
    if wal.exists():                            # a checkpointed database has no WAL, but copy it if it lingers
        shutil.copy2(wal, Path(str(part) + ".wal"))
    info = verify(part)
    part.replace(dest)
    print(f"backed up {db_path} -> {dest} ({info['bytes']:,} bytes, {info['tables']} tables, {info['rows']})")
    return dest


def restore(backup_path: Path, db_path: Path | None = None, *, force: bool = False) -> Path:
    """Replace the database with a backup, keeping the current file as a safety copy."""
    backup_path = Path(backup_path)
    db_path = Path(db_path or settings.duckdb_path)
    verify(backup_path)                          # never restore something unreadable

    if db_path.exists():
        try:
            import duckdb
            probe = duckdb.connect(str(db_path))   # fails fast if another process holds it
            probe.close()
        except Exception as exc:
            raise SystemExit(
                f"refusing to restore: {db_path} is in use ({exc}).\n"
                "Stop the app (Ctrl-C the uvicorn/`python main.py` process) and run this again."
            )

    if db_path.exists():
        safety = db_path.with_name(f"{db_path.stem}.pre-restore-{_stamp()}{db_path.suffix}")
        shutil.copy2(db_path, safety)
        print(f"current database kept at {safety}")
    for stale in (Path(str(db_path) + ".wal"),):
        if stale.exists() and not force:
            stale.rename(stale.with_name(stale.name + f".pre-restore-{_stamp()}"))
    shutil.copy2(backup_path, db_path)
    print(f"restored {backup_path} -> {db_path}")
    return db_path


def list_backups(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or DEFAULT_BACKUP_DIR)
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up, verify or restore the app's DuckDB file")
    parser.add_argument("--db", type=Path, default=settings.duckdb_path, help="database file (default: settings.duckdb_path)")
    parser.add_argument("--out", type=Path, default=DEFAULT_BACKUP_DIR, help="backup directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backup", help="checkpoint, copy and verify")
    b.add_argument("--label", default="", help="suffix for the file name (e.g. before-migration)")

    sub.add_parser("list", help="list backups newest first")

    v = sub.add_parser("verify", help="open a file read-only and report its contents")
    v.add_argument("file", type=Path)

    r = sub.add_parser("restore", help="replace the database with a backup (app must be stopped)")
    r.add_argument("file", type=Path)
    r.add_argument("--force", action="store_true", help="move a stale .wal aside without asking")

    args = parser.parse_args(argv)

    if args.cmd == "backup":
        path = backup(args.db, args.out, args.label)
        print(path)
    elif args.cmd == "list":
        rows = list_backups(args.out)
        if not rows:
            print(f"no backups in {args.out}")
        for p in rows:
            print(f"{p.name:44s} {p.stat().st_size / 1e6:8.1f} MB  {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")
    elif args.cmd == "verify":
        print(verify(args.file))
    elif args.cmd == "restore":
        restore(args.file, args.db, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
