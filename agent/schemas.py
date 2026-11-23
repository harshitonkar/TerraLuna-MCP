"""
TerraLuna - Pydantic Schemas for Autonomous Landing Zone Audit
Defines validated data models for landing zone telemetry,
tool traces, safety criteria, and mission control reports.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ToolTraceItem(BaseModel):
    """Execution telemetry record for an MCP Tool or Resource call."""
    tool_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    timestamp: str


class SlopeMetrics(BaseModel):
    """Terrain slope and roughness statistics."""
    mean_slope_deg: float
    max_slope_deg: float
    min_slope_deg: float
    std_slope_deg: float
    unsafe_slope_pct: float
    max_safe_slope_deg: float = 10.0
    mean_tri_m: float
    max_tri_m: float
    is_slope_safe: bool


class SolarMetrics(BaseModel):
    """Solar illumination and power viability statistics."""
    sun_elevation_deg: float
    sun_azimuth_deg: float
    sunlit_pct: float
    shadowed_pct: float
    min_required_pct: float = 65.0
    is_illumination_safe: bool
    is_permanently_shadowed_region: bool
    solar_power_viability: str


class RegolithMetrics(BaseModel):
    """Spectral regolith composition and mineral classification."""
    dataset_type: str
    mean_feo_wt_pct: Optional[float] = None
    mean_tio2_wt_pct: Optional[float] = None
    mean_hydration_index: Optional[float] = None
    regolith_classification: str
    isru_resource_potential: Optional[str] = None


class EarthAnalogComparison(BaseModel):
    """Quantitative fidelity match against Lonar Crater basalt testbed."""
    overall_analog_match_pct: float
    slope_similarity_pct: float
    roughness_similarity_pct: float
    mineral_affinity_pct: float
    earth_analog_site: str
    recommendation: str


class LandingZoneReport(BaseModel):
    """
    Final synthesized autonomous landing site validation report.
    Presents complete mission scorecard, deterministic engineering parameters,
    live MCP tool trace, and executive mission brief.
    """
    status: Literal["APPROVED", "REJECTED_HAZARD", "OUT_OF_BOUNDS"]
    safety_score: float = Field(ge=0.0, le=100.0, description="Overall landing safety score from 0 to 100")
    dataset_id: str
    target_coordinates: Dict[str, float]
    search_radius_m: int
    slope_metrics: Optional[SlopeMetrics] = None
    solar_metrics: Optional[SolarMetrics] = None
    regolith_summary: Optional[Dict[str, Any]] = None
    earth_analog_similarity: Optional[EarthAnalogComparison] = None
    hazard_flags: List[str] = Field(default_factory=list)
    mesh_artifact_path: Optional[str] = None
    illumination_artifact_path: Optional[str] = None
    tool_trace: List[ToolTraceItem] = Field(default_factory=list)
    execution_mode: str = Field(description="'Gemini LLM + MCP' or 'Offline Deterministic + MCP'")
    mission_brief: str
