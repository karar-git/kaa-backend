"""Planet physics: orbit size, speed, temperature, radius, distance.

Computed offline in section 13b of the pipeline and validated there against the
NASA Exoplanet Archive. As everywhere else in this API, nothing is calculated on
request -- these routes read `planet_physics.csv`, `physics_validation.csv` and
`planet_radius_measurements.csv`, and say which numbers are ours.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from .. import data
from ..schemas import PlanetCard, PlanetPhysics, ValidationReport

router = APIRouter(tags=["physics"])

CRITERIA = {
    "agrees": "|z| <= 2, where z is the difference over the combined uncertainty",
    "close": "|z| > 2 (or no uncertainty) but within 10% of the published value",
    "disagrees": "neither of the above",
    "inconclusive (below 3 sigma)": "our depth is below 3 sigma, so any agreement "
                                    "is agreement with noise and does not count",
    "null (star brightened)": "our depth is negative: no dip was measured",
}


def _resolve(target: str) -> str:
    """Accept our names (HATP-10), archive names (WASP-11) and planet names."""
    df = data.planet_physics()
    t = target.strip().lower()
    for col in ("target", "archive_host", "planet"):
        hit = df[df[col].str.lower() == t]
        if not hit.empty:
            return hit.target.iloc[0]
    raise HTTPException(404, f"Unknown planet '{target}'. See GET /api/planets.")


@router.get("/planets", response_model=list[PlanetPhysics],
            summary="How far, how fast, how hot: every planet")
def list_planets(
    sort: str = Query("distance_ly", description="Any numeric column, e.g. "
                      "orbital_speed_kms, a_au, teq_k, distance_ly"),
    descending: bool = Query(False),
) -> list[dict]:
    """One row per planet: orbit size, orbital speed, temperature, starlight
    received, and distance from Earth, each with an uncertainty.

    **Where the numbers come from.** Orbit size and speed follow from Kepler's
    third law using the archive's period and stellar mass; distance from Earth is
    the Gaia distance quoted by the archive, converted to light-years. Our frames
    cannot measure either -- a 6-inch telescope with no astrometric solution has no
    parallax -- and the `speed_distance_basis` and `distance_source` fields say so
    on every row. The radius from **our** measured depths is under
    `GET /api/planets/{target}`.

    Every value here is cross-checked against the archive's published number in
    `GET /api/planets/validation`.
    """
    df = data.planet_physics()
    if sort not in df.columns or not pd.api.types.is_numeric_dtype(df[sort]):
        raise HTTPException(422, f"Cannot sort by '{sort}'.")
    return data.to_records(df.sort_values(sort, ascending=not descending))


@router.get("/planets/validation", response_model=ValidationReport,
            summary="Derived values checked against published values")
def validation(
    uses_our_data: bool | None = Query(
        None, description="true: only checks built on our measured depths. "
                          "false: only equation checks built on archive inputs."),
    status: str | None = Query(None, examples=["disagrees"]),
    target: str | None = Query(None, examples=["CoRoT-2"]),
) -> dict:
    """Every derived quantity next to the NASA Exoplanet Archive value, with the
    percentage difference, a z-score and a status.

    Two kinds of check, reported separately because they test different things:

    * **Equation checks** (`uses_our_data=false`) rebuild archive values from other
      archive values: orbit size, a/R*, transit duration, temperature,
      insolation, radius. They test our equations and constants.
    * **Measurement checks** (`uses_our_data=true`) compare the depth we measured,
      and the radius it implies, with the published ones. They test our data.

    Disagreements are returned, not filtered out. Where an equation check
    disagrees, recomputing by hand from the archive's own neighbouring inputs
    reproduces our number: the archive's composite table mixes papers, and its
    published insolation or temperature is not always consistent with the radius,
    temperature and orbit it sits beside.
    """
    df = data.physics_validation()
    rows = df
    if uses_our_data is not None:
        rows = rows[rows.uses_our_data == uses_our_data]
    if status:
        rows = rows[rows.status == status]
    if target:
        rows = rows[rows.target == _resolve(target)]

    eq, ms = df[~df.uses_our_data], df[df.uses_our_data]
    med = (eq.assign(a=eq.pct_diff.abs()).groupby("quantity").a.median()
           .round(2).dropna().to_dict())
    solid = ms[ms.status.isin(["agrees", "close", "disagrees"])]
    n_field = int(solid.note.fillna("").str.contains("field candidate").sum())
    finding = (
        f"{int((eq.status == 'agrees').sum())} of {len(eq)} equation checks agree "
        f"within 2 sigma and {int((eq.status == 'disagrees').sum())} disagree. "
        f"Of {len(ms)} checks on our own depths, {len(solid)} rest on a depth of at "
        f"least 3 sigma, and {n_field} of those come from field stars whose identity "
        f"as the host is unverified. The rest are inconclusive or null, so this "
        f"dataset does not independently measure these planets' sizes.")
    return {
        "criteria": CRITERIA,
        "counts_equation_checks": eq.status.value_counts().to_dict(),
        "counts_measurement_checks": ms.status.value_counts().to_dict(),
        "median_abs_pct_diff_by_quantity": med,
        "finding": finding,
        "rows": data.to_records(rows),
    }


@router.get("/planets/equations", summary="Equations, constants and assumptions")
def equations() -> dict:
    """Everything needed to reproduce `/api/planets` by hand."""
    return {
        "equations": [
            {"quantity": "a_au", "name": "Kepler's third law",
             "formula": "a = cbrt( G (M_star + M_planet) P^2 / (4 pi^2) )",
             "inputs": ["pl_orbper", "st_mass", "pl_bmassj"]},
            {"quantity": "orbital_speed_kms", "name": "Circular orbital speed",
             "formula": "v = 2 pi a / P", "inputs": ["a", "pl_orbper"]},
            {"quantity": "a_over_rstar", "formula": "a / R_star",
             "inputs": ["a", "st_rad"]},
            {"quantity": "transit_duration_h", "name": "Total transit duration T14",
             "formula": "T14 = (P/pi) asin( sqrt((1+k)^2 - b^2) / ((a/R_star) sin i) ), "
                        "sin i = sqrt(1 - (b R_star/a)^2)",
             "inputs": ["pl_orbper", "pl_ratror (k)", "pl_imppar (b)", "a", "st_rad"]},
            {"quantity": "teq_k", "name": "Equilibrium temperature",
             "formula": "Teq = T_star sqrt(R_star / 2a)",
             "inputs": ["st_teff", "st_rad", "a"],
             "assumption": "Bond albedo 0, heat fully redistributed"},
            {"quantity": "insolation_earth", "name": "Incident flux",
             "formula": "S/S_earth = (R_star/R_sun)^2 (T_star/5772 K)^4 / (a/AU)^2",
             "inputs": ["st_rad", "st_teff", "a"]},
            {"quantity": "rp_rjup", "name": "Planet radius from transit depth",
             "formula": "R_p = R_star sqrt(depth)",
             "inputs": ["depth (ours, or pl_trandep)", "st_rad"],
             "assumption": "No limb darkening; biases R_p by a few percent"},
            {"quantity": "distance_ly", "name": "Distance",
             "formula": "d_ly = d_pc x 3.2616", "inputs": ["sy_dist (Gaia)"]},
        ],
        "constants": {
            "G": "6.67430e-11 m^3 kg^-1 s^-2 (CODATA 2018)",
            "GM_sun": "1.3271244e20 m^3 s^-2 (IAU 2015 nominal)",
            "GM_jup": "1.2668653e17 m^3 s^-2 (IAU 2015 nominal)",
            "R_sun": "6.957e8 m (IAU 2015 nominal)",
            "R_jup": "7.1492e7 m (IAU 2015 nominal, equatorial)",
            "AU": "1.495978707e11 m", "parsec": "3.0856775814913673e16 m",
            "light_year": "9.4607304725808e15 m", "T_sun": "5772 K",
        },
        "uncertainties": "Monte Carlo: 20,000 normal draws of every archive input with "
                         "its published error; median and 16-84% half-width reported. "
                         "Inputs with no published error are held fixed and listed in "
                         "`inputs_without_errors`.",
        "orbit_assumption": "Circular orbits. All eight archive eccentricities are "
                            "below 0.05.",
        "external_data_disclosure":
            "Inputs are from the NASA Exoplanet Archive `pscomppars` table (TAP), "
            "retrieved when the pipeline ran. Only transit depths are measured from "
            "our MicroObservatory frames.",
    }


@router.get("/planets/{target}", response_model=PlanetCard,
            summary="One planet: physics, our radius measurements, validation")
def planet(target: str = Path(..., examples=["CoRoT-2"],
                              description="HATP-10, WASP-11 and 'WASP-11 b' all work")
           ) -> dict:
    """Everything known about one planet in one response: the derived physics,
    every radius we could compute from our own depths (with its significance and
    status), every validation check, and the archive row it was all built from."""
    t = _resolve(target)
    phys = data.to_records(data.planet_physics().query("target == @t"))[0]
    radii = data.to_records(data.planet_radius_measurements().query("target == @t"))
    checks = data.to_records(data.physics_validation().query("target == @t"))
    cat = data.to_records(data.planet_catalog().query("target == @t"))[0]

    measured = [r for r in radii if r["status"] == "measured"
                and r["depth_source"].startswith("target window")]
    ours = (f"our best target-window radius is {measured[0]['rp_rjup']:.2f} ± "
            f"{measured[0]['rp_err_rjup']:.2f} R_jup against a published "
            f"{measured[0]['catalog_rp_rjup']:.2f}" if measured else
            "no night gave a target-window depth of 3 sigma or more, so we do not "
            "report a radius of our own")
    # A headline orbit that disagrees with the archive's own published one should
    # not reach a reader without that published value beside it.
    a_chk = [c for c in checks if c["quantity"] == "semi_major_axis"]
    a_note = ""
    if a_chk and a_chk[0]["pct_diff"] is not None and abs(a_chk[0]["pct_diff"]) > 10:
        a_note = (f", {a_chk[0]['pct_diff']:+.0f}% from the archive's published "
                  f"{a_chk[0]['published']:.4f} AU because its stellar-mass input is "
                  f"{cat['st_mass']} ± {cat['st_masserr1']} solar masses")
    summary = (f"{phys['planet']} circles its star every "
               f"{phys['orbital_period_days']:.2f} days at {phys['a_au']:.4f} AU "
               f"({phys['orbit_in_mercury_orbits']:.2f} of Mercury's orbit{a_note}), "
               f"moving at "
               f"{phys['orbital_speed_kms']:.0f} km/s, about "
               f"{phys['distance_ly']:.0f} light-years away; {ours}.")
    return {"physics": phys, "radius_measurements": radii, "validation": checks,
            "catalog_inputs": cat, "summary": summary}
