"""Layer specs are built once per catalog version, not once per request.

`derive_layer_specs` runs a COUNT(*) for every visible table - 740,679 crime rows on the
real database - and every map pan used to pay for it. The cache is keyed on
`schema.catalog_version()` and cleared explicitly after a data-only reload.
"""

from services import map_layers


def test_specs_are_built_once_per_catalog_version(reference_data, monkeypatch):
    calls = {"build": 0}
    real = map_layers._build_layer_specs

    def counted():
        calls["build"] += 1
        return real()

    monkeypatch.setattr(map_layers, "_build_layer_specs", counted)
    map_layers.invalidate_caches()

    first = map_layers.derive_layer_specs()
    second = map_layers.derive_layer_specs()
    assert calls["build"] == 1
    assert first == second


def test_invalidate_forces_a_rebuild(reference_data, monkeypatch):
    calls = {"build": 0}
    real = map_layers._build_layer_specs

    def counted():
        calls["build"] += 1
        return real()

    monkeypatch.setattr(map_layers, "_build_layer_specs", counted)
    map_layers.invalidate_caches()
    map_layers.derive_layer_specs()
    map_layers.invalidate_caches()  # what services/data_sources.refresh_source does
    map_layers.derive_layer_specs()
    assert calls["build"] == 2


def test_a_new_catalog_version_invalidates_the_cache(reference_data, monkeypatch):
    calls = {"build": 0}
    real = map_layers._build_layer_specs

    def counted():
        calls["build"] += 1
        return real()

    monkeypatch.setattr(map_layers, "_build_layer_specs", counted)
    versions = iter(["v1", "v1", "v2"])
    monkeypatch.setattr(map_layers.schema, "catalog_version", lambda: next(versions, "v2"))
    map_layers.invalidate_caches()

    map_layers.derive_layer_specs()
    map_layers.derive_layer_specs()
    map_layers.derive_layer_specs()  # catalog changed under the cache
    assert calls["build"] == 2


def test_callers_get_copies_so_a_panel_cannot_corrupt_the_cache(reference_data):
    map_layers.invalidate_caches()
    first = map_layers.derive_layer_specs()
    assert first, "the reference fixture should expose at least one layer"
    first[0]["title"] = "mutated by a caller"
    second = map_layers.derive_layer_specs()
    assert second[0]["title"] != "mutated by a caller"


def test_describe_layers_is_cached_too(reference_data, monkeypatch):
    calls = {"build": 0}
    real = map_layers._build_describe_layers

    def counted(version):
        calls["build"] += 1
        return real(version)

    monkeypatch.setattr(map_layers, "_build_describe_layers", counted)
    map_layers.invalidate_caches()

    payload = map_layers.describe_layers()
    again = map_layers.describe_layers()
    assert calls["build"] == 1
    assert payload["catalog_version"] == again["catalog_version"]
    assert payload["layers"][0]["name"] == "houses"  # the special, JS-driven layer
