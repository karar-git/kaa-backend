"""Response schemas.

These exist mainly so the generated OpenAPI document is genuinely useful: every
field carries a unit and a one-line description, because "flux" and "depth" mean
nothing to a reader without them. Tabular endpoints that mirror a results CSV
column-for-column return `list[dict]` rather than a frozen model, so adding a
column to the pipeline does not silently drop it from the API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Quality = Literal["good", "marginal", "unusable"]
Stretch = Literal["zscale", "asinh", "linear"]
SourceLabel = Literal["star", "hot_pixel", "cosmic_ray", "satellite_trail",
                      "faint_source", "noise", "unknown"]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    team: str
    challenge: str
    data_dir: str
    results_present: int = Field(description="How many expected result files exist")
    results_missing: list[str] = Field(description="Expected files that are absent")
    images_available: bool
    ai_enabled: bool = Field(
        description="True when OPENROUTER_API_KEY is set on the server")
    timing: dict[str, Any] = Field(
        description="Whether the `bjd_tdb` column in these results is genuinely "
                    "barycentric. The pipeline falls back to plain JD_UTC when "
                    "astropy.coordinates cannot be imported, which mislabels "
                    "every event time by up to 8 minutes. Check "
                    "`barycentric_correction_applied` before quoting a timing.")


class Instrument(BaseModel):
    telescope: str = Field(examples=["Cecilia"])
    aperture_mm: int = Field(examples=[150], description="Primary mirror diameter")
    focal_length_mm: int = Field(examples=[560])
    observatory: str
    latitude_deg: float
    longitude_deg: float
    height_m: float
    detector_width_px: int
    detector_height_px: int
    arcsec_per_px: float = Field(examples=[5.0])
    field_of_view_arcmin: list[float] = Field(description="[width, height]")
    bit_depth: int = Field(examples=[12], description="ADC range is 0-4095")
    exposure_s: float
    filters: list[str]


class DatasetMeta(BaseModel):
    science_frames: int
    dark_frames: int
    total_files: int
    brief_claims: int = Field(
        description="Record count stated in the challenge brief (~1633)")
    discrepancy: int = Field(
        description="Actual total minus the brief's figure. Flagged, not hidden.")
    targets: int
    sessions: int
    date_first: str
    date_last: str
    exptime_unique: list[float]
    filters_science: list[str]
    filters_dark: list[str]
    image_shape: list[int]
    arcsec_per_px: float
    camtemp_K_range: list[float]
    airmass_range: list[float]
    instrument: Instrument


class CalibrationSummary(BaseModel):
    n_darks: int
    master_median_counts: float
    master_spatial_sigma_counts: float
    single_dark_spatial_sigma_counts: float
    random_noise_1_dark: float = Field(
        description="Random noise a SINGLE dark would inject, in ADU")
    random_noise_full_stack: float = Field(
        description="Random noise left after combining all 60 darks")
    hot_pixels: int
    hot_pixel_fraction_pct: float
    fixed_pattern_stability_corr_30_nights: float = Field(
        description="Pixel-to-pixel correlation between the first and last dark, "
                    "30 nights apart. Near 1.0 means the pattern is stable and "
                    "therefore worth subtracting.")
    camtemp_K_range_darks: list[float]


class SessionSummary(BaseModel):
    session_id: str = Field(examples=["TRES-5__2026-08-26"])
    target: str
    night: str
    n_frames: int
    start_utc: str
    end_utc: str
    span_hours: float
    median_cadence_s: float | None
    quality: Quality | None = None
    median_contrast: float | None = Field(
        default=None,
        description="Median (brightest non-hot pixel - sky) / sky sigma. "
                    "The main usability discriminator.")
    median_sources: float | None = Field(
        default=None, description="Median count of pixels above sky + 5 sigma")
    median_weather: float | None = Field(
        default=None,
        description="Header WEATHER keyword. Empirically a transparency score "
                    "where 100 = clear (r = +0.75 against our own pixel metrics).")
    field_motion_px: float | None = Field(
        default=None, description="Total field drift across the session")


class TargetSummary(BaseModel):
    target: str
    nights: int
    good: int
    marginal: int
    unusable: int
    n_frames: int | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    usable: bool = Field(description="False when no night reached 'good'")


class LightCurvePoint(BaseModel):
    """One photometric measurement. Every field except `frame_index` is optional:
    a column the pipeline could not produce arrives as null rather than failing
    the whole response."""

    frame_index: int
    t_utc: str | None = None
    bjd_tdb: float | None = Field(
        default=None,
        description="Barycentric Julian Date (TDB). Corrected for Earth's orbital "
                    "position, which shifts event times by up to +/-8 minutes.")
    target_flux: float | None = Field(
        default=None, description="Sky-subtracted aperture sum, ADU")
    comp_flux_sum: float | None = None
    norm_flux: float | None = Field(
        default=None,
        description="Differential flux, normalised to the session median. "
                    "This is the transit curve.")
    target_only_norm: float | None = Field(
        default=None,
        description="Target flux alone, for comparison. Its larger scatter shows "
                    "what the comparison stars removed.")
    airmass: float | None = None
    sky_level: float | None = None
    WEATHER: float | None = None


class LightCurve(BaseModel):
    session_id: str
    target: str
    quality: Quality | None
    n_points: int
    rms_ppt: float | None = Field(
        description="Scatter of the differential curve, parts per thousand")
    rms_target_only_ppt: float | None
    improvement_factor: float | None = Field(
        description="How much the comparison stars reduced the scatter. Below ~1 "
                    "means they are not working and no dip can be trusted.")
    target_saturated: bool | None = Field(
        default=None,
        description="True when the chosen target hits the 4095 ADC ceiling")
    prediction: dict | None = Field(
        default=None, description="Predicted transit window from the ephemeris")
    points: list[LightCurvePoint]


class FalsePositiveCase(BaseModel):
    id: int
    name: str
    description: str
    signature: str
    test: str
    testable_with_our_data: bool
    endpoint: str | None = Field(
        default=None, description="API route holding the evidence for this test")


class ClassifyResult(BaseModel):
    label: SourceLabel
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    note: str | None = None


class ExplainRequest(BaseModel):
    session_id: str = Field(examples=["TRES-5__2026-08-26"])
    audience: Literal["public", "student", "astronomer"] = "student"
    language: Literal["en", "ar"] = "en"


class ExplainResponse(BaseModel):
    session_id: str
    language: str
    audience: str
    text: str
    model_name: str
    grounded_on: dict[str, Any] = Field(
        description="The exact measured numbers handed to the model. The model is "
                    "asked to phrase these, never to produce them.")


# ------------------------------------------------------------------ physics

ValidationStatus = Literal["agrees", "close", "disagrees", "not checkable",
                           "inconclusive (below 3 sigma)", "null (star brightened)"]


class PlanetPhysics(BaseModel):
    """How far, how fast, how hot -- one row per planet.

    Derived with textbook equations from NASA Exoplanet Archive inputs, with
    Monte Carlo uncertainties. None of these numbers come from our frames except
    where a field name says `ours`; see `speed_distance_basis`.
    """
    target: str = Field(examples=["CoRoT-2"], description="Target name used in this dataset")
    planet: str = Field(examples=["CoRoT-2 b"], description="Archive planet name")
    archive_host: str = Field(examples=["CoRoT-2"],
                              description="Archive host name. HATP-10 is filed as WASP-11.")
    orbital_period_days: float = Field(description="Archive orbital period (input)")
    a_au: float = Field(description="Orbit size (semi-major axis), AU. Kepler III: "
                                    "a^3 = G(M*+Mp)P^2/4pi^2")
    a_au_err: float | None = Field(description="1-sigma uncertainty, AU")
    orbit_in_mercury_orbits: float = Field(
        description="a divided by Mercury's orbit (0.387 AU). 0.07 means the planet "
                    "is ~14x closer to its star than Mercury is to the Sun.")
    orbit_circumference_million_km: float = Field(description="2 pi a, in million km")
    orbital_speed_kms: float = Field(description="Orbital speed v = 2 pi a / P, km/s")
    orbital_speed_kms_err: float | None = None
    orbital_speed_vs_earth: float = Field(
        description="Speed relative to Earth's 29.78 km/s around the Sun")
    a_over_rstar: float = Field(description="Orbit size in units of the star's radius")
    a_over_rstar_err: float | None = None
    transit_duration_h: float = Field(
        description="Predicted first-to-fourth contact duration, hours: "
                    "T14 = P/pi asin(sqrt((1+k)^2-b^2) / ((a/R*) sin i))")
    transit_duration_h_err: float | None = None
    teq_k: float = Field(description="Equilibrium temperature, K, albedo 0: "
                                     "Teq = T* sqrt(R*/2a)")
    teq_k_err: float | None = None
    insolation_earth: float = Field(
        description="Starlight received relative to Earth (Earth = 1)")
    insolation_earth_err: float | None = None
    rp_from_catalog_depth_rjup: float = Field(
        description="Planet radius from the ARCHIVE depth, Jupiter radii: R* sqrt(depth). "
                    "An equation check; for the radius from OUR depth see "
                    "/api/planets/{target} -> radius_measurements.")
    rp_from_catalog_depth_rjup_err: float | None = None
    distance_pc: float = Field(description="Distance from Earth, parsecs (Gaia, external)")
    distance_pc_err: float | None = None
    distance_ly: float = Field(description="Distance from Earth, light-years (Gaia, "
                                          "external). Also the age of the light we recorded.")
    distance_ly_err: float | None = None
    star_vmag: float | None = Field(description="Host star V magnitude. Larger is fainter.")
    header_pointing_offset_arcmin: float | None = Field(
        description="Separation between the FITS header pointing and the catalogued "
                    "host, after precessing the catalogue to the observing date.")
    header_pointing_offset_px: float | None = Field(description="Same, in pixels at 5\"/px")
    inputs_without_errors: str | None = Field(
        description="Archive inputs that had no published error and were held fixed. "
                    "Uncertainties are underestimated for these.")
    catalog_source: str = Field(description="Where and when the inputs were retrieved")
    distance_source: str
    speed_distance_basis: str


class ValidationCheck(BaseModel):
    """One derived number compared with the archive's published value."""
    target: str
    planet: str
    quantity: str = Field(examples=["semi_major_axis"])
    unit: str | None = None
    derived: float | None = Field(description="Our value")
    derived_err: float | None
    published: float | None = Field(description="NASA Exoplanet Archive value")
    published_err: float | None = Field(description="Null where the archive gives none")
    pct_diff: float | None = Field(description="100 (derived - published) / published")
    z_score: float | None = Field(
        description="Difference over the combined uncertainty. Null when neither side "
                    "has one.")
    status: ValidationStatus = Field(
        description="`agrees`: |z| <= 2. `close`: |z| > 2 but within 10%. "
                    "`disagrees`: neither. For checks on our own depth, `inconclusive` "
                    "and `null` override agreement, because agreement with noise is "
                    "not validation.")
    equation: str
    uses_our_data: bool = Field(
        description="True only for checks that use a depth measured from our frames")
    note: str | None = None


class RadiusMeasurement(BaseModel):
    """Planet radius from a transit depth measured in our frames."""
    target: str
    session_id: str
    depth_source: str = Field(
        description="`target window` = the star chosen by position, measured in the "
                    "predicted window. `best field candidate` = the most significant "
                    "star at the published phase that passed the field test; its "
                    "identity as the host is unverified.")
    track_id: int | None = None
    depth_pct: float
    depth_err_pct: float | None
    significance_sigma: float
    rp_rjup: float | None = Field(description="R* sqrt(depth), Jupiter radii. Null "
                                              "for a negative depth.")
    rp_err_rjup: float | None
    catalog_depth_pct: float | None
    catalog_rp_rjup: float | None
    depth_z_vs_catalog: float | None
    status: Literal["measured", "inconclusive (below 3 sigma)", "null (star brightened)"]


class PlanetCard(BaseModel):
    physics: PlanetPhysics
    radius_measurements: list[RadiusMeasurement]
    validation: list[ValidationCheck]
    catalog_inputs: dict[str, Any] = Field(
        description="The raw archive row every derived value above was computed from")
    summary: str = Field(description="One plain sentence, generated deterministically")


class ValidationReport(BaseModel):
    criteria: dict[str, str]
    counts_equation_checks: dict[str, int] = Field(
        description="Status counts for checks built only from archive inputs. These "
                    "test our equations.")
    counts_measurement_checks: dict[str, int] = Field(
        description="Status counts for checks that use our measured depths. These "
                    "test our data.")
    median_abs_pct_diff_by_quantity: dict[str, float]
    finding: str
    rows: list[ValidationCheck]


class KPIs(BaseModel):
    """Headline numbers for a dashboard landing page or a status report.

    Every value is read from a results file; nothing is estimated here. Each group
    names the endpoint holding the detail behind it.
    """
    dataset: dict[str, Any]
    data_quality: dict[str, Any]
    photometry: dict[str, Any]
    transit_search: dict[str, Any]
    planets: dict[str, Any]
    physics_validation: dict[str, Any]
    ai: dict[str, Any]
    generated_utc: str | None = Field(description="When the pipeline produced these results")
