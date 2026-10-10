"""services/census_metrics.py - the metro-geometry derived-measure provider."""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402

PIT = "Pittsburgh, PA Metro Area"
IND = "Indianapolis-Carmel-Anderson, IN Metro Area"
DEN = "Denver-Aurora-Lakewood, CO Metro Area"
SQ_KM_PER_SQ_MI = census_fixture.SQ_KM_PER_SQ_MI


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("metrics"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def provider(fx):
    from services import census_metrics
    census_metrics.reset_caches()
    return census_metrics.PROVIDER


def ents(*names):
    return [{"value": n} for n in names]


def test_areas_populations_and_density_match_the_known_geometry(fx, provider):
    frame = provider.compute(ents(PIT, IND), ["population", "land_area", "density"]).frame
    assert list(frame.columns) == ["msa_name", "population", "land_area_sq_mi", "population_density"]
    for _, row in frame.iterrows():
        exp = fx.expected[row["msa_name"]]
        assert row["population"] == exp["population"]
        assert row["land_area_sq_mi"] == pytest.approx(exp["area_sq_mi"], abs=0.006)     # rounded to 2 dp
        assert row["population_density"] == pytest.approx(exp["density"], abs=0.006)


def test_rows_follow_the_order_requested(fx, provider):
    assert list(provider.compute(ents(IND, PIT), ["land_area"]).frame["msa_name"]) == [IND, PIT]


def test_only_the_requested_measures_become_columns(fx, provider):
    assert list(provider.compute(ents(PIT), ["land_area"]).frame.columns) == ["msa_name", "land_area_sq_mi"]
    assert list(provider.compute(ents(PIT), ["population"]).frame.columns) == ["msa_name", "population"]


def test_new_engine_agrees_with_the_function_it_replaced(fx, provider):
    from services import map_layers
    new = provider.compute(ents(PIT, IND), ["population", "land_area", "density"]).frame
    old = map_layers.get_msa_population_density([PIT, IND], request="population density of Pittsburgh and Indianapolis")
    for new_col, old_col in (("population", "population"), ("land_area_sq_mi", "land_area_sq_mi"),
                             ("population_density", "population_density")):
        assert np.allclose(new[new_col].to_numpy(float), old[old_col].to_numpy(float), rtol=1e-4)


def test_square_kilometers_only_when_asked(fx, provider):
    mi = provider.compute(ents(PIT), ["land_area", "density"], request="land area of Pittsburgh").frame.iloc[0]
    km = provider.compute(ents(PIT), ["land_area", "density"], request="land area of Pittsburgh in square kilometers").frame.iloc[0]
    assert list(provider.compute(ents(PIT), ["land_area", "density"], request="sq km").frame.columns) == [
        "msa_name", "land_area_sq_km", "population_density_per_sq_km"]
    assert km["land_area_sq_km"] == pytest.approx(mi["land_area_sq_mi"] * SQ_KM_PER_SQ_MI, abs=0.01)
    assert km["population_density_per_sq_km"] == pytest.approx(mi["population_density"] / SQ_KM_PER_SQ_MI, abs=0.01)


@pytest.mark.parametrize("text,expected", [
    ("land area in square kilometers", True), ("area in sq km", True), ("km2 please", True), ("area in km\u00b2", True),
    ("a 5 km commute", False), ("land area in square miles", False), ("area of Pittsburgh", False), ("", False),
])
def test_unit_detection(text, expected):
    from services import census_metrics
    assert census_metrics.wants_km(text) is expected


def test_unresolved_cbsa_code_falls_back_to_the_title(fx, provider):
    result = provider.compute(ents("Beaver Valley Test, PA Metro Area"), ["land_area"])
    assert result.frame.iloc[0]["land_area_sq_mi"] == pytest.approx(fx.expected[PIT]["counties"]["Beaver County"]["area_sq_mi"], abs=0.006)
    assert any("unresolved" in n for n in result.notes)


def test_a_metro_without_county_membership_is_a_clear_data_error(fx, provider):
    from services.derived_measures import DerivedError
    with pytest.raises(DerivedError) as err:
        provider.compute(ents(DEN), ["land_area"])
    assert err.value.code == "data_unavailable" and "diagnose_msa" in err.value.message


def test_more_than_the_cap_is_declined_not_attempted(fx, provider):
    from services import census_metrics
    from services.derived_measures import DerivedError
    with pytest.raises(DerivedError) as err:
        provider.compute(ents(*[f"Metro {i}" for i in range(census_metrics.MAX_METROS + 1)]), ["land_area"])
    assert err.value.code == "limit_exceeded" and "name the metros" in err.value.message


def test_missing_geometry_is_reported_with_the_fix(fx, provider, monkeypatch):
    from services import geo_utils
    from services.derived_measures import DerivedError
    monkeypatch.setattr(geo_utils, "_load_tracts_gdf", lambda: None)
    with pytest.raises(DerivedError) as err:
        provider.compute(ents(PIT), ["land_area"])
    assert err.value.code == "data_unavailable" and "setup_data.py" in err.value.message


def test_status_codes_match_the_agent_layer():
    from agents import answer_status
    from services import census_metrics
    assert census_metrics.DATA_UNAVAILABLE == answer_status.DATA_UNAVAILABLE
    assert census_metrics.PLACE_MISSING == answer_status.PLACE_MISSING
    assert census_metrics.LIMIT_EXCEEDED == answer_status.LIMIT_EXCEEDED


# --------------------------------------------------------------------------------------------------------------------
# Coverage: tolerate a sliver of missing geometry, refuse to publish a wrong number
# --------------------------------------------------------------------------------------------------------------------
@contextmanager
def gap_metro(monkeypatch, *, tracts=100, with_geometry=99, missing_pop=1000):
    """A metro of ``tracts`` tracts (1000 people each) of which only ``with_geometry`` have a polygon."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    import db.duckdb_store as store
    from services import geo_utils

    conn = store.get_conn()
    name, code = "Gap Test, ZZ Metro Area", "55555"
    conn.execute("INSERT INTO census_msa (msa_code, geo_id, name, population) VALUES (?, 'g', ?, ?)", [code, name, tracts * 1000])
    conn.execute("INSERT INTO cbsa_counties (cbsa_code, cbsa_title, msa_type, state_fips, county_fips, county_name, state_name) "
                 "VALUES (?, 'Gap Test, ZZ', 'Metropolitan Statistical Area', '99', '999', 'Gap County', 'Zed')", [code])
    rows = []
    for i in range(tracts):
        pop = 1000 if i < with_geometry else missing_pop
        rows.append((f"99999{i:06d}", f"g{i}", f"T{i}", pop))
    conn.executemany("INSERT INTO census_tracts (tract_fips, geo_id, name, population) VALUES (?,?,?,?)", rows)
    base = geo_utils._load_tracts_gdf()
    extra = gpd.GeoDataFrame({"tract_fips": [r[0] for r in rows[:with_geometry]]},
                             geometry=[box(-100 + (i % 10) * 0.1, 35 + (i // 10) * 0.1, -99.95 + (i % 10) * 0.1, 35.05 + (i // 10) * 0.1)
                                       for i in range(with_geometry)], crs="EPSG:4326")
    merged = gpd.GeoDataFrame(pd.concat([base, extra], ignore_index=True), crs="EPSG:4326")
    monkeypatch.setattr(geo_utils, "_load_tracts_gdf", lambda: merged)
    try:
        yield name
    finally:
        conn.execute("DELETE FROM census_tracts WHERE substr(tract_fips, 1, 5) = '99999'")
        conn.execute("DELETE FROM cbsa_counties WHERE cbsa_code = ?", [code])
        conn.execute("DELETE FROM census_msa WHERE msa_code = ?", [code])


def test_a_sliver_of_missing_geometry_is_tolerated_and_reported(fx, provider, monkeypatch):
    with gap_metro(monkeypatch, tracts=100, with_geometry=99) as name:
        result = provider.compute(ents(name), ["population", "land_area", "density"])
    assert result.frame.iloc[0]["population"] == 99_000          # footprint only: figures always reconcile
    assert any("1 of 100 census tracts had no boundary geometry" in n for n in result.notes)


def test_too_little_geometry_is_refused_not_guessed(fx, provider, monkeypatch):
    from services.derived_measures import DerivedError
    with gap_metro(monkeypatch, tracts=100, with_geometry=90) as name:
        with pytest.raises(DerivedError) as err:
            provider.compute(ents(name), ["land_area"])
    assert err.value.code == "data_unavailable" and "different years" in err.value.message


def test_many_empty_tracts_without_geometry_do_not_block_an_answer(fx, provider, monkeypatch):
    """4 of 100 tracts (tract coverage 96%) hold almost nobody: area is still trustworthy, and it says so."""
    with gap_metro(monkeypatch, tracts=100, with_geometry=96, missing_pop=1) as name:
        result = provider.compute(ents(name), ["land_area"])
    assert any("4 of 100" in n for n in result.notes)


def test_population_that_disagrees_with_the_msa_file_is_reconciled_in_a_note(fx, provider):
    import db.duckdb_store as store
    conn = store.get_conn()
    conn.execute("UPDATE census_msa SET population = population * 1.1 WHERE name = ?", [PIT])
    try:
        result = provider.compute(ents(PIT), ["population"])
    finally:
        conn.execute("UPDATE census_msa SET population = ? WHERE name = ?", [fx.expected[PIT]["population"], PIT])
    assert any("Census MSA file lists" in n for n in result.notes)
    assert not provider.compute(ents(PIT), ["population"]).notes          # agreeing data -> no noise


# --------------------------------------------------------------------------------------------------------------------
# Caching and concurrency
# --------------------------------------------------------------------------------------------------------------------
def test_each_tract_is_measured_once_ever(fx, provider, monkeypatch):
    import geopandas as gpd
    calls = []
    real = gpd.GeoDataFrame.to_crs
    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", lambda self, *a, **k: (calls.append(len(self)), real(self, *a, **k))[1])
    provider.compute(ents(PIT), ["land_area"])
    first = len(calls)
    assert first >= 1
    provider.compute(ents(PIT), ["land_area", "density"])           # same tracts: no reprojection
    assert len(calls) == first
    provider.compute(ents(PIT, IND), ["land_area"])                  # only the NEW metro's tracts are measured
    assert len(calls) == first + 1 and calls[-1] == 6


def test_reloading_the_geometry_can_never_serve_stale_areas(fx, provider):
    import geopandas as gpd
    from config import settings
    before = provider.compute(ents(PIT), ["land_area"]).frame.iloc[0]["land_area_sq_mi"]
    cache = Path(settings.nri_shp).parent / "nri_geometry_cache.parquet"
    original = gpd.read_parquet(cache)
    try:
        doubled = original.copy()
        doubled["geometry"] = doubled.geometry.scale(xfact=2.0, yfact=1.0, origin="center")      # change every polygon
        doubled.to_parquet(cache)
        census_fixture._clear_geometry_caches()
        after = provider.compute(ents(PIT), ["land_area"]).frame.iloc[0]["land_area_sq_mi"]
    finally:
        original.to_parquet(cache)
        census_fixture._clear_geometry_caches()
    assert after > before * 1.5
    assert provider.compute(ents(PIT), ["land_area"]).frame.iloc[0]["land_area_sq_mi"] == pytest.approx(before)


def test_concurrent_requests_agree_and_do_not_crash(fx, provider):
    results, errors = [], []

    def work(i):
        try:
            names = [PIT, IND] if i % 2 else [IND, PIT]
            results.append(tuple(provider.compute(ents(*names), ["land_area"]).frame.set_index("msa_name")["land_area_sq_mi"].sort_index()))
        except Exception as exc:                                       # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(results)) == 1 and len(results) == 12
