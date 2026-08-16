"""Lightweight checks for BikePGH city-key normalization and routing helpers."""
from services.bike_routing import _in_pittsburgh_bounds
assert _in_pittsburgh_bounds(40.44, -80.01)
print("BikePGH routing helper checks passed.")
