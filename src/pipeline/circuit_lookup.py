"""Maps FastF1 EventName strings to the canonical circuit_key used in
data/circuit_archetypes.csv. Keyed on event name (not Location) because
Location drifts across seasons for the same physical circuit (Monaco/Monte
Carlo, Silverstone country label, Yas Island/Yas Marina) and because FastF1's
2026 schedule has a known bad row: "Bahrain Grand Prix" with Location
"Kuala Lumpur" (round 16). Event name is the stable key; Location is ignored
for archetype lookup entirely.

Spanish Grand Prix is special-cased: Madrid joined the 2026 calendar
alongside (not instead of) Barcelona, so from 2026 "Spanish Grand Prix"
maps to Madrid while "Barcelona Grand Prix" is its own separate round.
"""

EVENT_TO_CIRCUIT_KEY = {
    "70th Anniversary Grand Prix": "silverstone",
    "Abu Dhabi Grand Prix": "yas_marina",
    "Australian Grand Prix": "melbourne",
    "Austrian Grand Prix": "spielberg",
    "Azerbaijan Grand Prix": "baku",
    "Bahrain Grand Prix": "sakhir",
    "Barcelona Grand Prix": "barcelona",
    "Belgian Grand Prix": "spa",
    "Brazilian Grand Prix": "sao_paulo",
    "British Grand Prix": "silverstone",
    "Canadian Grand Prix": "montreal",
    "Chinese Grand Prix": "shanghai",
    "Dutch Grand Prix": "zandvoort",
    "Eifel Grand Prix": "nurburgring",
    "Emilia Romagna Grand Prix": "imola",
    "French Grand Prix": "le_castellet",
    "German Grand Prix": "hockenheim",
    "Hungarian Grand Prix": "budapest",
    "Italian Grand Prix": "monza",
    "Japanese Grand Prix": "suzuka",
    "Las Vegas Grand Prix": "las_vegas",
    "Mexican Grand Prix": "mexico_city",
    "Mexico City Grand Prix": "mexico_city",
    "Miami Grand Prix": "miami",
    "Monaco Grand Prix": "monaco",
    "Portuguese Grand Prix": "portimao",
    "Qatar Grand Prix": "lusail",
    "Russian Grand Prix": "sochi",
    "Sakhir Grand Prix": "sakhir",
    "Saudi Arabian Grand Prix": "jeddah",
    "Singapore Grand Prix": "marina_bay",
    "Styrian Grand Prix": "spielberg",
    "São Paulo Grand Prix": "sao_paulo",
    "Turkish Grand Prix": "istanbul",
    "Tuscan Grand Prix": "mugello",
    "United States Grand Prix": "austin",
    # "Spanish Grand Prix" handled separately below (year-dependent)
}


def event_to_circuit_key(event_name: str, season: int) -> str:
    if event_name == "Spanish Grand Prix":
        return "madrid" if season >= 2026 else "barcelona"
    key = EVENT_TO_CIRCUIT_KEY.get(event_name)
    if key is None:
        raise KeyError(
            f"No circuit_key mapping for event {event_name!r}. "
            "Add it to EVENT_TO_CIRCUIT_KEY in circuit_lookup.py."
        )
    return key
