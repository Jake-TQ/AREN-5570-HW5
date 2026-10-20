"""Preliminary feeder/main-breaker estimator for 3-phase, 4-wire panels.

This is a classroom load model, not a NEC-compliant design tool. Supply actual
nameplate VA, demand factors, conductor ampacity, panel rating, and listed
breaker interrupting ratings for a real installation.
"""
import argparse
from builtins import dict, float, str, int, bool, ValueError, tuple
import json
import math
from dataclasses import asdict, dataclass
from typing import Any
from dataclasses import dataclass
from math import ceil, sqrt


SYSTEMS = {
    "480Y/277": 480,  # line-to-line volts; 277 V line-to-neutral
    "208Y/120": 208,  # line-to-line volts; 120 V line-to-neutral
}
# Standard trip ratings listed in the user's table.
STANDARD_BREAKERS_A = (50, 100, 150, 200, 225, 300, 400, 600, 800,
                       1200, 2000, 2400, 3000, 4000)

# Frame size, minimum trip, maximum trip, instantaneous ratings, short-time
# ratings. Ratings are amperes and are transcribed from the user-provided table.
BREAKER_FRAMES = (
    (600, 50, 600, {"208/240": 42_000, "480": 30_000}, {"208/240": 30_000, "480": 30_000}),
    (800, 50, 800, {"208/240": 42_000, "480": 30_000}, {"208/240": 30_000, "480": 30_000}),
    (1600, 50, 1600, {"208/240": 65_000, "480": 50_000}, {"208/240": 50_000, "480": 50_000}),
    (2000, 50, 2000, {"208/240": 65_000, "480": 50_000}, {"208/240": 50_000, "480": 50_000}),
    (3000, 1200, 3000, {"208/240": 85_000, "480": 65_000}, {"208/240": 65_000, "480": 65_000}),
    (4000, 1600, 4000, {"208/240": 130_000, "480": 85_000}, {"208/240": 85_000, "480": 85_000}),
)


@dataclass(frozen=True)
class BreakerRecommendation:
    system: str
    connected_load_va: float
    calculated_load_va: float
    load_current_a: float
    minimum_breaker_a: float
    breaker_rating_a: int
    frame_size_a: int
    available_fault_current_a: float | None
    breaker_interrupting_rating_a: int
    panel_sccr_a: int


def size_panel_breaker(
    *,
    system: str,
    fixture_count: int,
    receptacle_count: int,
    fixture_va: float,
    panel_sccr_a: int | None = None,
    symmetrical_fault_current_a: float | None = None,
    receptacle_va: float = 180.0,
    trip_mode: str = "instantaneous",
    continuous_load_factor: float = 1.25,
    noncontinuous_load_factor: float = 1.0,
    lighting_is_continuous: bool = True,
    panel_rating_a: int | None = None,
) -> BreakerRecommendation:
    """Estimate a 3-pole panel breaker from aggregate VA and optional fault current.

    The connected load inputs are per fixture and per receptacle in volt-amperes.
    The model applies the selected factors to the lighting and receptacle totals,
    then assumes a balanced 3-phase load: I = VA / (sqrt(3) * line-line volts).
    If fault current is omitted, frame selection still uses the smallest frame
    that supports the selected trip rating, but fault withstand is not checked.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {tuple(SYSTEMS)}")
    if fixture_count < 0 or receptacle_count < 0:
        raise ValueError("fixture_count and receptacle_count cannot be negative")
    if fixture_va < 0 or receptacle_va < 0:
        raise ValueError("per-device VA values cannot be negative")
    if symmetrical_fault_current_a is not None and symmetrical_fault_current_a < 0:
        raise ValueError("fault current must be nonnegative")
    if panel_sccr_a is not None and panel_sccr_a <= 0:
        raise ValueError("panel SCCR must be positive")
    if continuous_load_factor < 1 or noncontinuous_load_factor < 1:
        raise ValueError("load factors must be at least 1.0")
    if trip_mode not in ("instantaneous", "short-time"):
        raise ValueError("trip_mode must be 'instantaneous' or 'short-time'")

    lighting_va = fixture_count * fixture_va
    receptacles_va = receptacle_count * receptacle_va
    connected_va = lighting_va + receptacles_va
    lighting_factor = continuous_load_factor if lighting_is_continuous else noncontinuous_load_factor
    receptacle_demand_va = min(receptacles_va, 10_000) + 0.50 * max(receptacles_va - 10_000, 0)
    calculated_va = lighting_va * lighting_factor + receptacle_demand_va
    load_current = calculated_va / (sqrt(3) * SYSTEMS[system])
    minimum_breaker = load_current  # Must also satisfy conductor and equipment limits.

    breaker_options = [a for a in STANDARD_BREAKERS_A if a >= minimum_breaker]
    if panel_rating_a is not None:
        breaker_options = [a for a in breaker_options if a <= panel_rating_a]
    if not breaker_options:
        raise ValueError("No standard breaker rating fits the calculated load/panel rating")
    breaker_a = breaker_options[0]

    frame = next((row for row in BREAKER_FRAMES if row[1] <= breaker_a <= row[2]), None)
    if frame is None:
        raise ValueError("Calculated breaker rating is outside the supplied frame table")
    frame_size, _, _, instantaneous, short_time = frame
    voltage_key = "208/240" if system == "208Y/120" else "480"
    interrupting_a = (instantaneous if trip_mode == "instantaneous" else short_time)[voltage_key]
    if (symmetrical_fault_current_a is not None
            and symmetrical_fault_current_a > interrupting_a):
        raise ValueError(
            f"Available fault current ({symmetrical_fault_current_a:g} A) exceeds "
            f"the {interrupting_a:,} A {trip_mode} rating for the {frame_size}-A frame"
        )
    if (symmetrical_fault_current_a is not None and panel_sccr_a is not None
            and panel_sccr_a < symmetrical_fault_current_a):
        raise ValueError("Panel SCCR is below the available fault current")

    return BreakerRecommendation(
        system=system,
        connected_load_va=connected_va,
        calculated_load_va=calculated_va,
        load_current_a=load_current,
        minimum_breaker_a=minimum_breaker,
        breaker_rating_a=breaker_a,
        frame_size_a=frame_size,
        available_fault_current_a=symmetrical_fault_current_a,
        breaker_interrupting_rating_a=interrupting_a,
        panel_sccr_a=panel_sccr_a,
    )


def size_lighting_and_receptacle_panels(
    *,
    fixture_count: int,
    receptacle_count: int,
    fixture_va: float,
    lighting_fault_current_a: float | None = None,
    receptacle_fault_current_a: float | None = None,
    lighting_panel_sccr_a: int | None = None,
    receptacle_panel_sccr_a: int | None = None,
    receptacle_va: float = 180.0,
    trip_mode: str = "instantaneous",
) -> dict[str, BreakerRecommendation]:
    """Size both panels: lighting at 480Y/277 V and receptacles at 208Y/120 V."""
    lighting = size_panel_breaker(
        system="480Y/277",
        fixture_count=fixture_count,
        receptacle_count=0,
        fixture_va=fixture_va,
        symmetrical_fault_current_a=lighting_fault_current_a,
        panel_sccr_a=lighting_panel_sccr_a,
        trip_mode=trip_mode,
    )
    receptacles = size_panel_breaker(
        system="208Y/120",
        fixture_count=0,
        receptacle_count=receptacle_count,
        fixture_va=0,
        symmetrical_fault_current_a=receptacle_fault_current_a,
        panel_sccr_a=receptacle_panel_sccr_a,
        receptacle_va=receptacle_va,
        lighting_is_continuous=False,
        trip_mode=trip_mode,
    )
    return {"lighting_panel": lighting, "receptacle_panel": receptacles}


if __name__ == "__main__":
    results = size_lighting_and_receptacle_panels(
        fixture_count=100,
        receptacle_count=200,
        fixture_va=52 / 0.95,
    )
    for panel_name, result in results.items():
        print(f"{panel_name}: Frame Size ({result.frame_size_a} AF)/ "
              f"Trip Size ({result.breaker_rating_a} AT)")

