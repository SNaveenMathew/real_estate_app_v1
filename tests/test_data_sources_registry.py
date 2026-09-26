"""
Structural contract tests for the services/data_sources.py registry.

These exist so adding a new built-in data source can't silently skip the
validation/backup pattern the rest of the registry follows — per the Copilot
review's caveat that a new source "should follow the same validation/backup
pattern," this makes that a test failure instead of a code-review hope.
"""
import pytest

from services import data_sources as ds
from services.crime_sources import CRIME_PARSERS

REGISTRY = ds.REGISTRY

# Tables more than one file/city writes into. A loader for one of these can
# silently skip a single bad file while that file's *old* rows sit untouched
# under its filename — which a bare post-load row count can't tell apart from a
# genuine success (see refresh_source()'s docstring). Any "append"-placement
# source backed by one of these MUST declare a `precheck`.
SHARED_MUTABLE_TABLES = {"houses", "sold_homes", "crime_incidents"}


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_source_is_internally_consistent(key):
    source = REGISTRY[key]
    assert source.key == key
    assert source.label.strip()
    assert source.category.strip()
    assert source.table.strip()
    assert source.instructions.strip()
    assert source.accept.strip()
    assert source.placement in ("single", "append")
    assert callable(source.loader)
    assert source.row_count_sql.strip()
    assert source.source_url == "" or source.source_url.startswith(("http://", "https://")), (
        f"{key!r}'s source_url doesn't look like a URL: {source.source_url!r}"
    )


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_append_sources_on_a_shared_table_declare_a_precheck(key):
    """
    This is the test that would have caught the Copilot review's two safety
    findings before they shipped (the CBSA no-primary-key issue aside, which is
    a single-file source and caught by a different mechanism — see
    refresh_source). bike is deliberately exempt: each upload is scoped to its
    own shapefile's exact relative path, so a stale row can only ever be one
    with that identical path — already unambiguous without a precheck, the same
    way the single-file sources are.
    """
    source = REGISTRY[key]
    if source.placement == "append" and source.table in SHARED_MUTABLE_TABLES:
        assert source.precheck is not None, (
            f"{key!r} writes into the shared table {source.table!r} as an append source "
            "without a `precheck`. A malformed same-named replacement could leave stale "
            "rows in place while still reporting success. Add one (see "
            "_precheck_redfin_file / _precheck_sold_file / _make_crime_precheck for the "
            "pattern) before shipping this source."
        )


def test_precheck_is_callable_and_returns_the_expected_shape(tmp_path):
    """Every declared precheck actually runs and returns (bool, str) — catches a
    typo'd signature or an exception on the simplest possible input (an empty
    file) rather than that surfacing for the first time on someone's real upload."""
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    for key, source in REGISTRY.items():
        if source.precheck is None:
            continue
        ok, reason = source.precheck(empty)
        assert isinstance(ok, bool), f"{key!r}'s precheck didn't return a bool as its first value"
        assert isinstance(reason, str), f"{key!r}'s precheck didn't return a str as its second value"
        assert ok is False, f"{key!r}'s precheck accepted a completely empty file"


def test_registry_crime_keys_match_crime_city_parsers():
    """services/data_sources.py's crime registry and services/crime_sources.py's
    CRIME_PARSERS are two separate lists that have to stay in lockstep — this
    catches either one drifting from the other (a parser added without a
    matching data source, or vice versa)."""
    crime_keys = {f"crime_{p.city}" for p in CRIME_PARSERS}
    registered_crime_keys = {k for k in REGISTRY if k.startswith("crime_")}
    assert crime_keys == registered_crime_keys


def test_direct_download_sources_have_no_leftover_navigation_language():
    """A source marked direct_download=True is supposed to BE the file. If its
    instructions still tell the user to go find/navigate to a download, that's
    either a copy-paste leftover from before it was upgraded or a sign
    direct_download was set without updating the copy — either way, wrong."""
    banned_phrases = ("portal page", "not a direct file", "you will need to find", "navigate")
    for key, source in REGISTRY.items():
        if not source.direct_download:
            continue
        lowered = source.instructions.lower()
        hit = next((p for p in banned_phrases if p in lowered), None)
        assert hit is None, (
            f"{key!r} is direct_download=True but its instructions still say {hit!r}: "
            f"{source.instructions!r}"
        )
