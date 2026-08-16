
"""Regression checks for BikePGH visualization semantics.

Run from the app root after installing the usual requirements:
    python tests/test_bikepg_h_visualization.py
"""
from pathlib import Path
import importlib.util

EXPECTED = {
    "bike_lanes": ("Bike Lanes", "steelblue"),
    "bikeable_sidewalks": ("Bikeable Sidewalks", "lightblue"),
    "cautionary_bike_route": ("Cautionary Bike Route", "red"),
    "on_street_bike_route": ("On Street Bike Route", "lightgreen"),
    "protected_bike_lanes": ("Protected Bike Lanes", "darkgreen"),
    "sharrows": ("Sharrows", "orange"),
    "trails": ("Trails", "pink"),
}

def load_data_loader():
    spec = importlib.util.spec_from_file_location(
        "data_loader", Path(__file__).parents[1] / "services" / "data_loader.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    mod = load_data_loader()
    specs = getattr(mod, "_BIKE_LAYER_SPECS")
    assert specs, "Bike layer specs are empty"

    normalized = {k: v for k, v in specs.items()}
    for folder, (key, label, color) in normalized.items():
        assert EXPECTED[key] == (label, color), (folder, key, label, color)
    print("PASS: BikePGH standard labels/colors match the ground-truth R specification.")

if __name__ == "__main__":
    main()
