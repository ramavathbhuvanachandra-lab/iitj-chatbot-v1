import json
from pathlib import Path


# Path to data/campus_locations.json
DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "campus_locations.json"
)


def load_locations():
    """Load all campus locations from JSON."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Campus Navigation] Error loading locations: {e}")
        return []


def find_location(query: str):
    """
    Find a campus location by name or alias.
    Supports both exact match and questions like:
    - Where is the library?
    - How do I reach the Central Mess?
    """

    query = query.strip().lower()

    locations = load_locations()

    for location in locations:

        # Match location name anywhere in the question
        name = location.get("name", "").lower()
        if name and name in query:
            return location

        # Match aliases anywhere in the question
        for alias in location.get("aliases", []):
            alias = alias.lower()
            if alias in query:
                return location

    return None