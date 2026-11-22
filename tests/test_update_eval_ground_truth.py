"""The fixture-maintenance script has invariants of its own.

update_eval_ground_truth.py rewrites constants inside eval/golden_set.py from a freshly built
fixture. Nothing tested it, so a bug there would either corrupt the golden set silently or
leave it stale against the fixture it claims to describe. These tests pin the helper that does
the rewriting and check that the checked-in golden set is a fixed point of the script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "update_eval_ground_truth.py"
GOLDEN = ROOT / "eval" / "golden_set.py"


@pytest.fixture(scope="module")
def updater():
    spec = importlib.util.spec_from_file_location("update_eval_ground_truth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_named_query_is_read_only(updater):
    """Read-only in intent: SELECT, or a WITH ... SELECT chain. Nothing here writes."""
    for name, sql in updater.SQL.items():
        assert sql.strip().lower().startswith(("select", "with")), name


def test_replace_assignment_touches_only_the_named_variable(updater):
    text = "A = 1\nB = 2\nA_SUFFIX = 3\n"
    out = updater.replace_assignment(text, "A", 10)
    assert out == "A = 10\nB = 2\nA_SUFFIX = 3\n"


def test_replace_assignment_refuses_a_missing_variable(updater):
    with pytest.raises(RuntimeError):
        updater.replace_assignment("B = 2\n", "A", 1)


def test_normalize_number_keeps_integral_values_integral(updater):
    assert updater.normalize_number(4.0) == 4
    assert isinstance(updater.normalize_number(4.0), int)
    assert updater.normalize_number(4.5) == 4.5
    assert updater.normalize_number(None) is None


def test_first_number_requires_exactly_one_row(updater):
    assert updater.first_number([{"v": 2.0}], "v") == 2
    with pytest.raises(RuntimeError):
        updater.first_number([{"v": 1}, {"v": 2}], "v")


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    """The value build_truth derives from a freshly built fixture database."""
    import duckdb
    from eval.fixtures import build_fixture

    path = tmp_path_factory.mktemp("updater") / "fixture.duckdb"
    build_fixture(path)
    conn = duckdb.connect(str(path))
    try:
        from importlib import util as importlib_util

        spec = importlib_util.spec_from_file_location("update_eval_ground_truth_truth", SCRIPT)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.build_truth(conn)
    finally:
        conn.close()


def test_the_checked_in_golden_set_is_a_fixed_point_of_the_updater(updater, truth):
    text = GOLDEN.read_text(encoding="utf-8")
    patched = updater.patch_golden(text, truth)
    assert patched == text, "eval/golden_set.py is stale against eval/fixtures.py; run the updater"
