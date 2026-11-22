"""
Unit and integration tests for the Federal Transit Administration (FTA) datasets integration:
- Database schema and row counts (transit_agencies, transit_ridership, transit_safety, transit_routes)
- Catalog definitions, relationships, and multi-tier joins
- Map layers auto-discovery and bounding-box queries
- Multimodal transit shortest path routing engine
- API endpoint POST /api/transit/route
- Agent tool find_transit_route
"""
import json
import pytest
from fastapi.testclient import TestClient

import db.duckdb_store as store
from services.map_layers import describe_layers, get_lines
from services.transit_routing import route_transit
from api.common import TransitRouteRequest
from agents.tools import find_transit_route
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestTransitDatabase:
    """Test physical tables and data integrity in DuckDB."""

    def test_tables_exist_and_populated(self):
        agencies_count = store.query("SELECT COUNT(*) AS c FROM transit_agencies").iloc[0]["c"]
        ridership_count = store.query("SELECT COUNT(*) AS c FROM transit_ridership").iloc[0]["c"]
        safety_count = store.query("SELECT COUNT(*) AS c FROM transit_safety").iloc[0]["c"]
        routes_count = store.query("SELECT COUNT(*) AS c FROM transit_routes").iloc[0]["c"]

        assert agencies_count >= 2000, f"Expected >= 2000 agencies, got {agencies_count}"
        assert ridership_count >= 300000, f"Expected >= 300000 ridership records, got {ridership_count}"
        assert safety_count >= 20000, f"Expected >= 20000 safety records, got {safety_count}"
        assert routes_count >= 70000, f"Expected >= 70000 route features, got {routes_count}"

    def test_cbsa_and_msa_linkage(self):
        """Verify multi-tier join linkage from transit_agencies to census_msa."""
        df = store.query("""
            SELECT a.ntd_id, a.agency, a.msa_code, m.name AS msa_name
            FROM transit_agencies a
            JOIN census_msa m ON a.msa_code = m.msa_code
            WHERE a.msa_code IS NOT NULL
            LIMIT 10
        """)
        assert len(df) == 10
        assert "msa_name" in df.columns
        assert df["msa_name"].notnull().all()

    def test_ridership_aggregation(self):
        """Verify ridership queries by agency and year."""
        df = store.query("""
            SELECT agency, year, SUM(upt) AS total_ridership
            FROM transit_ridership
            WHERE agency ILIKE '%Transit%' OR agency ILIKE '%Authority%'
            GROUP BY agency, year
            ORDER BY total_ridership DESC
            LIMIT 5
        """)
        assert not df.empty
        assert (df["total_ridership"] > 0).all()

    def test_safety_aggregation(self):
        """Verify safety time-series queries by mode."""
        df = store.query("""
            SELECT mode, SUM(event_total) AS total_events, SUM(total_fatalities) AS total_fatalities
            FROM transit_safety
            GROUP BY mode
            ORDER BY total_events DESC
            LIMIT 5
        """)
        assert not df.empty
        assert (df["total_events"] > 0).all()


class TestTransitCatalog:
    """Test catalog metadata, ontology, and relationships."""

    def test_catalog_tables_registered(self):
        tables = store.query("""
            SELECT name FROM catalog_tables
            WHERE name IN ('transit_agencies', 'transit_ridership', 'transit_safety', 'transit_routes')
        """)["name"].tolist()
        expected = {'transit_agencies', 'transit_ridership', 'transit_safety', 'transit_routes'}
        assert expected.issubset(set(tables))

    def test_catalog_relationships_registered(self):
        rels = store.query("""
            SELECT left_table, right_table FROM catalog_relationships
            WHERE left_table LIKE 'transit_%' OR right_table LIKE 'transit_%'
        """)
        assert len(rels) >= 5
        rel_pairs = set(zip(rels["left_table"], rels["right_table"]))
        assert ("transit_routes", "transit_agencies") in rel_pairs
        assert ("transit_ridership", "transit_agencies") in rel_pairs
        assert ("transit_safety", "transit_agencies") in rel_pairs
        assert ("transit_agencies", "census_msa") in rel_pairs

    def test_catalog_glossary_concepts(self):
        concepts = store.query("""
            SELECT key FROM catalog_concepts
            WHERE key LIKE 'transit_%'
        """)["key"].tolist()
        assert "transit_ridership" in concepts
        assert "transit_safety_events" in concepts
        assert "transit_routes" in concepts


class TestTransitMapLayers:
    """Test map layers discovery and spatial GeoJSON retrieval."""

    def test_layer_discovery(self):
        layers_info = describe_layers()
        layers = layers_info.get("layers", [])
        transit_layer = next((l for l in layers if l["name"] == "transit_routes"), None)
        assert transit_layer is not None, "transit_routes not discovered in map layers"
        assert transit_layer["kind"] == "lines"
        assert transit_layer["title"] == "Transit Routes"

    def test_layer_lines_geojson(self):
        # Pittsburgh bounding box
        west, south, east, north = -80.05, 40.40, -79.90, 40.50
        fc = get_lines("transit_routes", west, south, east, north)
        assert fc is not None
        assert fc.get("type") == "FeatureCollection"
        features = fc.get("features", [])
        assert len(features) > 0, "Expected transit route features in Pittsburgh bounding box"
        first = features[0]
        assert first["geometry"]["type"] in ("LineString", "MultiLineString")
        assert "route_short_name" in first["properties"]
        assert "route_color" in first["properties"]


class TestTransitRouting:
    """Test transit routing engine."""

    @pytest.mark.asyncio
    async def test_route_transit_pittsburgh(self):
        res = await route_transit("Shadyside", "Highland Park", city="Pittsburgh, PA")
        assert res is not None
        assert res.get("provider") == "FTA National Transit Map"
        route = res.get("route") or {}
        summary = route.get("summary") or {}
        assert summary.get("length", 0) > 0.5
        assert summary.get("duration_minutes", 0) > 5.0
        assert len(route.get("shape", [])) >= 2
        assert len(route.get("maneuvers", [])) >= 2
        assert len(res.get("transit_lines", [])) >= 1

    @pytest.mark.asyncio
    async def test_route_transit_invalid_endpoints(self):
        with pytest.raises(ValueError):
            await route_transit("CompletelyUnknownPlaceXYZ123", "Shadyside")


class TestTransitAPIAndTool:
    """Test FastAPI endpoint and Agent tool."""

    def test_api_transit_route_endpoint(self, client):
        resp = client.post(
            "/api/transit/route",
            json={"start": "Shadyside", "end": "Highland Park", "city": "Pittsburgh, PA"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "route" in data
        assert data["route"]["summary"]["length"] > 0
        assert len(data["transit_lines"]) > 0

    def test_agent_tool_find_transit_route(self):
        raw = find_transit_route.func(start="Shadyside", end="Highland Park", city="Pittsburgh, PA")
        data = json.loads(raw)
        assert data.get("status") == "ok"
        assert data.get("kind") == "route_found"
        assert data.get("distance_miles", 0) > 0
        assert len(data.get("transit_lines_used", [])) > 0
