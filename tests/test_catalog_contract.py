"""Catalog contract: built-in catalog lint with a ratchet (see db/catalog_lint.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "golden" / "catalog_lint_baseline.json"


@pytest.fixture(scope="module")
def issues(tmp_path_factory):
    fx = census_fixture.activate(tmp_path_factory.mktemp("lint"))
    from db import catalog_lint
    yield catalog_lint.lint_catalog()
    census_fixture.deactivate(fx)


def test_the_built_in_catalog_has_no_lint_errors(issues):
    errors = [str(i) for i in issues if i.severity == "error"]
    assert not errors, "\n".join(errors)


def test_no_new_warnings_beyond_the_accepted_baseline(issues):
    accepted = set(json.loads(BASELINE.read_text())["accepted"])
    new = [str(i) for i in issues if i.severity == "warning" and i.key() not in accepted]
    assert not new, ("New catalog lint warnings (fix them, or - if deliberate - add the key to "
                     "tests/golden/catalog_lint_baseline.json in the same change):\n" + "\n".join(new))


def test_the_baseline_has_no_stale_entries(issues):
    """The ratchet only tightens: once a pre-existing warning is fixed it must leave the baseline."""
    accepted = set(json.loads(BASELINE.read_text())["accepted"])
    stale = sorted(accepted - {i.key() for i in issues})
    assert not stale, "Fixed warnings still listed in the baseline - remove them:\n" + "\n".join(stale)


def test_census_concepts_follow_the_alias_hygiene_rules(issues):
    mine = {"census_place_population", "census_land_area", "census_profile", "msa_counties", "census_unloaded_topics"}
    offenders = [str(i) for i in issues if i.code in {"generic_alias", "alias_collision"} and i.subject.split(":")[0] in mine]
    assert not offenders, "\n".join(offenders)
