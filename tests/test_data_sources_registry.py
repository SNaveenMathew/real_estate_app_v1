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
            f"{key!r} is direct_download=True but its instructions still say {hit!r}: {source.instructions!r}"
        )


def test_every_source_has_a_captured_source_link():
    missing = [
        key
        for key, source in REGISTRY.items()
        if not source.source_url.startswith(("http://", "https://"))
    ]
    assert not missing, f"data sources without a source URL: {missing}"


@pytest.mark.parametrize(
    ("key", "table"),
    [
        ("fta_ridership", "transit_ridership"),
        ("fta_safety", "transit_safety"),
        ("fta_routes", "transit_routes"),
    ],
)
def test_fta_sources_are_registered_and_uploadable(key, table):
    source = REGISTRY[key]
    assert source.table == table
    assert source.placement == "single"
    assert source.accept
    assert source.source_url.startswith("https://")
    assert callable(source.loader)


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("transit_ridership", "fta_ridership"),
        ("transit_safety", "fta_safety"),
        ("transit_routes", "fta_routes"),
        ("transit_route_zctas", "fta_routes"),
        ("transit_route_places", "fta_routes"),
    ],
)
def test_fta_sources_appear_on_their_catalog_tables(table, key):
    assert key in ds.sources_for_table(table)


def test_transit_agencies_points_at_the_ridership_workbook():
    assert "fta_ridership" in ds.sources_for_table("transit_agencies")


def test_fta_tables_are_on_a_domain_the_data_page_renders():
    from db.catalog_seed import SEED_DOMAINS
    from db.schema_catalog import DOMAIN_LABELS

    for table in (
        "transit_agencies",
        "transit_ridership",
        "transit_safety",
        "transit_routes",
        "transit_route_zctas",
        "transit_route_places",
    ):
        assert SEED_DOMAINS[table] in DOMAIN_LABELS
        assert SEED_DOMAINS[table] == "mobility"


def test_fta_workbook_upload_is_staged_under_the_loader_name(tmp_path, monkeypatch):
    from dataclasses import replace

    from config import settings

    canonical = tmp_path / "canonical-ridership.xlsx"
    monkeypatch.setattr(settings, "fta_ridership_file", canonical)
    source = replace(REGISTRY["fta_ridership"], dest_dir=tmp_path)
    upload = tmp_path / "September 2026.xlsx"
    upload.write_bytes(b"xlsx")

    dest, backups, new_files = ds._place_single(source, upload, upload.name)

    assert dest == canonical and canonical.read_bytes() == b"xlsx"
    assert new_files == [canonical]
    assert backups == {}


def test_fta_routes_zip_is_staged_as_the_loader_shapefile(tmp_path):
    import zipfile

    release = tmp_path / "release"
    release.mkdir()
    for suffix in ("shp", "dbf", "shx"):
        (release / f"NTM_Routes_2026.{suffix}").write_bytes(suffix.encode())
    archive = tmp_path / "NTM_Routes_2026.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in release.iterdir():
            zf.write(path, path.name)
    dest_dir = tmp_path / "connections"
    dest_dir.mkdir()

    shp, written = ds._extract_fta_routes_zip(
        archive, dest_dir, dest_dir / "National_Transit_Map_Routes.shp"
    )

    assert shp == dest_dir / "National_Transit_Map_Routes.shp"
    assert (dest_dir / "National_Transit_Map_Routes.dbf").exists()
    assert len(written) == 3


def test_fta_route_uid_is_stable_across_refreshes():
    from services.loaders.transit import _route_uid

    assert _route_uid("00001", "feed", "R1", "S1") == _route_uid("00001", "feed", "R1", "S1")
    assert _route_uid("00001", "feed", "R1", "S1") != _route_uid("00001", "feed", "R1", "S2")
    # Feeds can publish several segments with the same IDs; geometry keeps them distinct.
    assert _route_uid("00001", "feed", "R1", "S1", "line-a") != _route_uid(
        "00001", "feed", "R1", "S1", "line-b"
    )
    assert _route_uid("00001", "feed", "R1", "S1", "line-a", '{"name":"A"}') != _route_uid(
        "00001", "feed", "R1", "S1", "line-a", '{"name":"B"}'
    )
