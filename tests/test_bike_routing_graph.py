"""Routing correctness on a hand-built graph.

The routing engine's answers used to be covered only by integration tests against the live
BikePGH network: they could show that *a* route came back, never that the route was the
shortest one, that an unconnected network is reported as such, or that crime avoidance
reroutes instead of merely deleting edges. These tests build a four-node graph with known
weights, so the expected answer is arithmetic.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, box

from services import bike_routing as br

A = (-80.000, 40.440)
B = (-79.999, 40.440)
C = (-79.998, 40.440)
D = (-79.999, 40.441)  # detour node above B


def _graph(edges, geometries=None):
    """A graph shaped like _load_bike_graph's output, with weights given in miles."""
    adjacency = {}
    for a, b, miles in edges:
        adjacency.setdefault(a, {})[b] = miles
        adjacency.setdefault(b, {})[a] = miles
    edge_geometries = (
        geometries if geometries is not None else {tuple(sorted((a, b))): LineString([a, b]) for a, b, _miles in edges}
    )
    return {
        "adjacency": adjacency,
        "edge_geometries": edge_geometries,
        "source_records": [],
        "feature_count": len(edges),
        "city": "pittsburgh",
    }


def test_dijkstra_returns_the_cheapest_path_not_the_first_one():
    # A-B-C costs 2 miles; A-D-C costs 6. A search that returns on first arrival answers
    # 6 whenever it explores D before B, and nothing would notice.
    graph = _graph([(A, B, 1.0), (B, C, 1.0), (A, D, 3.0), (D, C, 3.0)])
    path, miles = br._dijkstra(graph, A, C)
    assert path == [A, B, C]
    assert miles == pytest.approx(2.0)


def test_dijkstra_is_symmetric():
    graph = _graph([(A, B, 1.0), (B, C, 1.0), (A, D, 3.0), (D, C, 3.0)])
    forward, forward_miles = br._dijkstra(graph, A, C)
    backward, backward_miles = br._dijkstra(graph, C, A)
    assert forward == list(reversed(backward))
    assert forward_miles == pytest.approx(backward_miles)


def test_dijkstra_refuses_to_invent_a_route_across_a_gap():
    # Two components: A-B and C-D. There is no path, and "no path" must be an error the
    # caller turns into no_route=True, never a straight line drawn between the endpoints.
    graph = _graph([(A, B, 1.0), (C, D, 1.0)])
    with pytest.raises(ValueError, match="No continuous path"):
        br._dijkstra(graph, A, D)


def test_crime_avoidance_without_hotspots_blocks_nothing():
    graph = _graph([(A, B, 1.0), (B, C, 1.0)])
    out = br._apply_crime_avoidance_filter(graph, [])
    assert out["crime_avoidance"]["applied"] is True
    assert out["crime_avoidance"]["blocked_edges"] == 0
    assert out["crime_avoidance"]["blocked_segments"] == 0
    assert len(out["adjacency"][A]) == 1


def test_crime_avoidance_blocks_the_hot_segment_and_reroutes_around_it():
    # A and B are junctions (each carries a dead-end stub), so the graph is cut into
    # intersection-to-intersection segments and the polygon over A-B blocks only that segment.
    # The direct route A-B-C is then unavailable and the answer must be the longer A-D-C,
    # with the removed geometry reported for the map.
    stub_a, stub_b = (-80.000, 40.4395), (-79.999, 40.4395)
    graph = _graph([(A, B, 1.0), (B, C, 1.0), (A, D, 3.0), (D, C, 3.0), (A, stub_a, 0.1), (B, stub_b, 0.1)])
    # Strictly between the junctions: a polygon that touched A or B would sit on every segment
    # that meets there, which is not the case being tested.
    hotspot = box(A[0] + 0.0004, A[1] - 0.0001, B[0] - 0.0004, B[1] + 0.0001)
    out = br._apply_crime_avoidance_filter(
        graph,
        [{"glon": B[0], "glat": B[1], "intensity": 9.0, "exclusion_polygon": hotspot, "crime_threshold": 1.0}],
        # Supply the density grid: otherwise the filter queries crime_incidents, which would make a
        # pure graph test depend on the live database being unlocked.
        all_density_cells=[{"glon": B[0], "glat": B[1], "intensity": 9.0}],
    )
    report = out["crime_avoidance"]
    assert report["blocked_edges"] >= 1
    assert report["blocked_segments"] >= 1
    assert B not in out["adjacency"].get(A, {})  # direct edge is gone
    path, miles = br._dijkstra(out, A, C)
    assert path == [A, D, C]
    assert miles == pytest.approx(6.0)
    assert report["blocked_network"]["features"], "the blocked geometry must be reported for the map"


def test_crime_avoidance_can_disconnect_the_graph():
    # Blocking the only corridor must leave no path at all - the honest answer - rather than
    # a route through crime because a fallback restored the edges.
    graph = _graph([(A, B, 1.0), (B, C, 1.0)])
    hotspot = box(A[0] - 0.01, A[1] - 0.01, C[0] + 0.01, C[1] + 0.01)
    out = br._apply_crime_avoidance_filter(
        graph,
        [{"glon": B[0], "glat": B[1], "intensity": 9.0, "exclusion_polygon": hotspot, "crime_threshold": 1.0}],
        # Supply the density grid: otherwise the filter queries crime_incidents, which would make a
        # pure graph test depend on the live database being unlocked.
        all_density_cells=[{"glon": B[0], "glat": B[1], "intensity": 9.0}],
    )
    with pytest.raises(ValueError, match="No continuous path"):
        br._dijkstra(out, A, C)


def test_snapping_splits_the_nearest_edge_and_keeps_its_length():
    graph = _graph([(A, C, 5.0)], geometries={tuple(sorted((A, C))): LineString([A, C])})
    node, miles = br._snap_point_to_graph(graph, lat=40.4400, lon=-79.9990)
    assert miles < 0.01
    assert node in graph["adjacency"]
    assert set(graph["adjacency"][A]) == {node}
    split = graph["adjacency"][A][node] + graph["adjacency"][node][C]
    real_length = br._line_length_miles(LineString([A, C]))
    assert split == pytest.approx(real_length, rel=0.02)  # the split keeps the real edge length
    assert tuple(sorted((A, C))) not in graph["edge_geometries"]


def test_snapping_refuses_an_endpoint_far_from_the_network():
    graph = _graph([(A, C, 5.0)])
    with pytest.raises(ValueError, match="nearest BikePGH path"):
        br._snap_point_to_graph(graph, lat=41.5, lon=-75.0)  # hundreds of miles away
