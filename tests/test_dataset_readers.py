"""Generic readers: same conventions as services/data_loader.py, identifiers keep their leading zeros."""
import json
import zipfile

import pandas as pd
import pytest

from services import dataset_readers as R


def test_csv_keeps_identifier_zeros_reads_bom_and_infers_types(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes('\ufefftract,zip,price,active,when,update\n01001020100,02134,"$1,250.50",yes,2024-05-01,x\n'
                  '42003040100,15213,900,no,2024-06-02,y\n'.encode("utf-8"))
    df, recs = R.normalize_frame(R.read_dataset(p).df)
    types = {r["name"]: r["dtype"] for r in recs}
    assert df["tract"][0] == "01001020100" and df["zip"][0] == "02134"
    assert types == {"tract": "text", "zip": "text", "price": "float", "active": "boolean", "when_col": "datetime", "update_col": "text"}
    assert df["price"][0] == 1250.5
    assert [r["source_name"] for r in recs][-2:] == ["when", "update"]      # the original header is kept


def test_latin1_fallback_and_semicolons(tmp_path):
    p = tmp_path / "b.csv"
    p.write_bytes("name;value\nCafé;1\nZoë;2\n".encode("latin-1"))
    res = R.read_dataset(p)
    assert any("latin-1" in n for n in res.notes) and res.df["name"][0] == "Café" and list(res.df.columns) == ["name", "value"]


def test_excel_header_row_is_found_below_title_rows(tmp_path):
    p = tmp_path / "c.xlsx"
    pd.DataFrame([["Census delineation file", None, None], ["Source: Census Bureau", None, None],
                  ["CBSA Code", "CBSA Title", "County FIPS"], ["38300", "Pittsburgh, PA", "003"],
                  ["19100", "Dallas, TX", "113"]]).to_excel(p, header=False, index=False)
    res = R.read_dataset(p)
    assert list(res.df.columns) == ["CBSA Code", "CBSA Title", "County FIPS"]
    assert any("row 3" in n for n in res.notes) and res.sheets
    df, _ = R.normalize_frame(res.df)
    assert df["cbsa_code"][0] == "38300" and df["county_fips"][0] == "003"


def test_json_jsonl_and_parquet(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"meta": 1, "rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}))
    (tmp_path / "b.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(tmp_path / "c.parquet")
    assert len(R.read_dataset(tmp_path / "a.json").df) == 2
    assert len(R.read_dataset(tmp_path / "b.jsonl").df) == 2
    assert list(R.read_dataset(tmp_path / "c.parquet").df.columns) == ["a", "b"]


def test_geojson_points_become_lat_lon_and_geometry(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "A"}, "geometry": {"type": "Point", "coordinates": [-80.0, 40.44]}},
        {"type": "Feature", "properties": {"name": "B"}, "geometry": {"type": "Point", "coordinates": [-79.9, 40.5]}}]}
    (tmp_path / "pts.json").write_text(json.dumps(fc))          # .json that is really GeoJSON is sniffed
    res = R.read_dataset(tmp_path / "pts.json")
    assert res.format == "geo" and res.geometry["points_only"]
    assert {"lat", "lon", "geometry_json", "min_lon"} <= set(res.df.columns)
    assert res.df["lon"].iloc[0] == pytest.approx(-80.0) and json.loads(res.df["geometry_json"].iloc[0])["type"] == "Point"


def test_zipped_shapefile_is_reprojected_to_wgs84(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    g = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(500000, 4500000), Point(500100, 4500100)], crs="EPSG:32617")
    folder = tmp_path / "shp"
    folder.mkdir()
    g.to_file(folder / "pts.shp")
    z = tmp_path / "pts.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in folder.iterdir():
            zf.write(f, f.name)
    res = R.read_dataset(z)
    assert any("Reprojected" in n for n in res.notes) and -85 < res.df["lon"].iloc[0] < -75 and 35 < res.df["lat"].iloc[0] < 45


def test_unsupported_and_empty_inputs_give_readable_errors(tmp_path):
    (tmp_path / "x.docx").write_text("nope")
    (tmp_path / "e.csv").write_text("")
    with pytest.raises(R.DatasetReadError, match="not supported"):
        R.read_dataset(tmp_path / "x.docx")
    with pytest.raises(R.DatasetReadError):
        R.read_dataset(tmp_path / "e.csv")


def test_numeric_cleaning_works_on_pandas_str_dtype():
    """data_loader._clean_numeric only acts on dtype==object; pandas 3 text columns are 'str'. The helper casts first."""
    s = pd.Series(["$1,234.5", "2", None])
    assert R._clean_numeric_series(s).dropna().tolist() == [1234.5, 2.0]


def test_sanitised_identifiers_pass_the_real_sql_guards_and_are_unique():
    from agents.tools import _FORBIDDEN
    taken: set[str] = set()
    names = [R.sanitize_identifier(n, taken) for n in ["Update", "LOAD", "% Lead", "Lead", "Copy", "call", "2020 pop", "Ünïcode Name"]]
    assert names[:3] == ["update_col", "load_col", "lead"] and names[3] == "lead_2" and len(set(names)) == len(names)
    assert names[6] == "c_2020_pop"
    for n in names:
        assert not _FORBIDDEN.search(f"SELECT {n} FROM t")
    assert R.slugify_table_name("EPA Smart Location Database (2021)") == "epa_smart_location_database_2021"
