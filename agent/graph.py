"""
TerraLuna - Autonomous AI Agent & LangGraph State Machine
Orchestrates FastMCP geospatial tools for autonomous landing zone audits.
Supports dual-mode execution:
1. Google Gemini LLM Mode (gemini-2.5-flash via langchain-google-genai)
2. 100% Offline Deterministic Fallback Mode (zero-key offline execution)
"""

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from mcp import StdioServerParameters

from agent.schemas import (
    EarthAnalogComparison,
    LandingZoneReport,
    RegolithMetrics,
    SlopeMetrics,
    SolarMetrics,
    ToolTraceItem,
)
from mcp_server.server import (
    DATASET_REGISTRY,
    LANDER_CONSTRAINTS,
    analyze_terrain_hazards,
    compare_lunar_earth_analog,
    extract_regolith_composition,
    get_lander_constraints,
    get_mission_catalog,
    simulate_solar_illumination,
)

# Load environment variables
load_dotenv(override=True)


class AgentState(TypedDict):
    """LangGraph agent state dictionary."""
    dataset_id: str
    lat: float
    lon: float
    radius_m: int
    sun_elevation_deg: float
    sun_azimuth_deg: float
    max_safe_slope_deg: float

    # Execution tracking
    is_valid_bounds: bool
    out_of_bounds_reason: Optional[str]
    terrain_result: Optional[Dict[str, Any]]
    solar_result: Optional[Dict[str, Any]]
    regolith_result: Optional[Dict[str, Any]]
    analog_result: Optional[Dict[str, Any]]

    status: str
    safety_score: float
    hazard_flags: List[str]
    tool_trace: List[Dict[str, Any]]
    execution_mode: str
    mission_brief: str
    final_report: Optional[Dict[str, Any]]


def get_mcp_server_params() -> StdioServerParameters:
    """
    Spawns the MCP stdio server subprocess using sys.executable.
    Dynamically resolves to the active environment's Python interpreter
    (e.g., /usr/local/bin/python on Linux / Streamlit Community Cloud and
    python.exe on Windows), avoiding any hardcoded Windows paths.
    """
    server_script = str(Path(__file__).resolve().parent.parent / "mcp_server" / "server.py")
    return StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ),
    )


def _record_tool_call(
    trace_list: List[Dict[str, Any]],
    tool_name: str,
    inputs: Dict[str, Any],
    func,
    *args,
    **kwargs,
) -> Any:
    """Executes a tool/function and logs its telemetry to the tool trace."""
    start_t = time.perf_counter()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        output = func(*args, **kwargs)
        duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        # Create compact summary without huge arrays
        if isinstance(output, dict):
            summary = {k: v for k, v in output.items() if not isinstance(v, list) or len(v) <= 5}
        else:
            summary = {"result": str(output)[:200]}

        trace_list.append({
            "tool_name": tool_name,
            "inputs": inputs,
            "output_summary": summary,
            "duration_ms": duration_ms,
            "timestamp": ts,
        })
        return output
    except Exception as e:
        duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        trace_list.append({
            "tool_name": tool_name,
            "inputs": inputs,
            "output_summary": {"error": str(e)},
            "duration_ms": duration_ms,
            "timestamp": ts,
        })
        raise e


# ==========================================
# LangGraph State Machine Nodes
# ==========================================

def node_validate_bounds(state: AgentState) -> Dict[str, Any]:
    """Node 1: Validates spatial coordinates against dataset bounds."""
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]
    tool_trace = list(state.get("tool_trace", []))

    # Read constraints resource
    _record_tool_call(
        tool_trace,
        "resource://twin/constraints/lander-specs",
        {},
        get_lander_constraints,
    )

    if dataset_id not in DATASET_REGISTRY:
        return {
            "is_valid_bounds": False,
            "out_of_bounds_reason": f"Unknown dataset_id '{dataset_id}'",
            "status": "OUT_OF_BOUNDS",
            "safety_score": 0.0,
            "hazard_flags": [f"Unknown dataset_id '{dataset_id}'"],
            "tool_trace": tool_trace,
        }

    b = DATASET_REGISTRY[dataset_id]["bounds"]
    in_bounds = (b["lat_min"] <= lat <= b["lat_max"]) and (b["lon_min"] <= lon <= b["lon_max"])

    if not in_bounds:
        reason = (
            f"Coordinates (lat: {lat:.4f}°, lon: {lon:.4f}°) are OUT OF BOUNDS for {dataset_id}. "
            f"Valid range: lat [{b['lat_min']:.4f}° to {b['lat_max']:.4f}°], "
            f"lon [{b['lon_min']:.4f}° to {b['lon_max']:.4f}°]."
        )
        return {
            "is_valid_bounds": False,
            "out_of_bounds_reason": reason,
            "status": "OUT_OF_BOUNDS",
            "safety_score": 0.0,
            "hazard_flags": ["OUT_OF_BOUNDS: Target coordinates outside mapped satellite raster extent"],
            "tool_trace": tool_trace,
        }

    return {
        "is_valid_bounds": True,
        "out_of_bounds_reason": None,
        "tool_trace": tool_trace,
    }


def node_analyze_terrain(state: AgentState) -> Dict[str, Any]:
    """Node 2: Executes Horn's slope and TRI terrain hazard analysis."""
    tool_trace = list(state.get("tool_trace", []))
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]
    radius_m = state.get("radius_m", 250)
    max_safe_slope = state.get("max_safe_slope_deg", 10.0)

    terrain_res = _record_tool_call(
        tool_trace,
        "tool://analyze_terrain_hazards",
        {"dataset_id": dataset_id, "lat": lat, "lon": lon, "radius_m": radius_m},
        analyze_terrain_hazards,
        dataset_id=dataset_id,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
    )

    hazard_flags = list(state.get("hazard_flags", []))
    if not terrain_res.get("is_slope_safe", False):
        hazard_flags.extend(terrain_res.get("hazard_reasons", ["Unsafe terrain slope"]))

    return {
        "terrain_result": terrain_res,
        "hazard_flags": hazard_flags,
        "tool_trace": tool_trace,
    }


def node_check_illumination(state: AgentState) -> Dict[str, Any]:
    """Node 3: Executes ray-cast solar horizon shadow analysis."""
    tool_trace = list(state.get("tool_trace", []))
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]
    sun_el = state.get("sun_elevation_deg", 12.0)
    sun_az = state.get("sun_azimuth_deg", 135.0)

    solar_res = _record_tool_call(
        tool_trace,
        "tool://simulate_solar_illumination",
        {"dataset_id": dataset_id, "lat": lat, "lon": lon, "sun_el": sun_el, "sun_az": sun_az},
        simulate_solar_illumination,
        dataset_id=dataset_id,
        lat=lat,
        lon=lon,
        sun_elevation_deg=sun_el,
        sun_azimuth_deg=sun_az,
    )

    hazard_flags = list(state.get("hazard_flags", []))
    if not solar_res.get("is_illumination_safe", False):
        hazard_flags.append(
            f"Insufficient solar illumination ({solar_res.get('sunlit_pct', 0):.1f}% < 65% minimum threshold)"
        )
    if solar_res.get("is_permanently_shadowed_region", False):
        hazard_flags.append("CRITICAL: Landing target situated in Permanently Shadowed Region (PSR) cold trap")

    return {
        "solar_result": solar_res,
        "hazard_flags": hazard_flags,
        "tool_trace": tool_trace,
    }


def node_analyze_regolith(state: AgentState) -> Dict[str, Any]:
    """Node 4: Extracts spectral regolith composition and hydration."""
    tool_trace = list(state.get("tool_trace", []))
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]
    radius_m = state.get("radius_m", 250)

    regolith_res = _record_tool_call(
        tool_trace,
        "tool://extract_regolith_composition",
        {"dataset_id": dataset_id, "lat": lat, "lon": lon, "radius_m": radius_m},
        extract_regolith_composition,
        dataset_id=dataset_id,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
    )

    return {
        "regolith_result": regolith_res,
        "tool_trace": tool_trace,
    }


def node_compare_earth_analog(state: AgentState) -> Dict[str, Any]:
    """Node 5: Benchmarks landing target against Lonar Crater basalt testbed."""
    tool_trace = list(state.get("tool_trace", []))
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]

    if dataset_id == "lunar_south_pole":
        analog_res = _record_tool_call(
            tool_trace,
            "tool://compare_lunar_earth_analog",
            {"lunar_lat": lat, "lunar_lon": lon, "earth_dataset": "lonar_analog"},
            compare_lunar_earth_analog,
            lunar_lat=lat,
            lunar_lon=lon,
        )
    else:
        analog_res = {
            "overall_analog_match_pct": 100.0,
            "slope_similarity_pct": 100.0,
            "roughness_similarity_pct": 100.0,
            "mineral_affinity_pct": 100.0,
            "earth_analog_site": "Lonar Meteorite Impact Crater (Self-Reference)",
            "recommendation": "DIRECT_EARTH_ANALOG: Target is the terrestrial testing baseline.",
        }

    return {
        "analog_result": analog_res,
        "tool_trace": tool_trace,
    }


def node_synthesize_report(state: AgentState) -> Dict[str, Any]:
    """Node 6: Synthesizes final LandingZoneReport with LLM or Deterministic Fallback brief."""
    is_valid = state.get("is_valid_bounds", True)
    dataset_id = state.get("dataset_id", "lunar_south_pole")
    lat = state["lat"]
    lon = state["lon"]
    radius_m = state.get("radius_m", 250)
    hazard_flags = list(state.get("hazard_flags", []))
    tool_trace_raw = state.get("tool_trace", [])

    if not is_valid:
        # Out-of-bounds termination
        brief = (
            f"### MISSION CONTROL ALERT: TARGET OUT OF BOUNDS\n\n"
            f"**Validation Verdict**: `OUT_OF_BOUNDS`\n"
            f"**Target Site**: Latitude {lat:.4f}°, Longitude {lon:.4f}°\n\n"
            f"**Reasoning**: {state.get('out_of_bounds_reason', 'Target coordinates fall outside planetary raster extent.')}\n"
            f"**Safety Action**: Automated mission abort safeguard engaged. No descent burn authorized."
        )
        report = LandingZoneReport(
            status="OUT_OF_BOUNDS",
            safety_score=0.0,
            dataset_id=dataset_id,
            target_coordinates={"lat": lat, "lon": lon},
            search_radius_m=radius_m,
            hazard_flags=hazard_flags,
            tool_trace=[ToolTraceItem(**t) for t in tool_trace_raw],
            execution_mode="Offline Deterministic + MCP",
            mission_brief=brief,
        )
        return {
            "status": "OUT_OF_BOUNDS",
            "safety_score": 0.0,
            "mission_brief": brief,
            "final_report": report.model_dump(),
        }

    terrain = state.get("terrain_result", {})
    solar = state.get("solar_result", {})
    regolith = state.get("regolith_result", {})
    analog = state.get("analog_result", {})

    # Compute deterministic safety score (0 - 100)
    mean_slope = terrain.get("mean_slope_deg", 5.0)
    unsafe_pct = terrain.get("unsafe_slope_pct", 0.0)
    mean_tri = terrain.get("mean_tri_m", 0.2)
    illum_pct = solar.get("sunlit_pct", 100.0)
    is_psr = solar.get("is_permanently_shadowed_region", False)
    max_safe_slope = state.get("max_safe_slope_deg", 10.0)

    score = 100.0
    if mean_slope > max_safe_slope:
        score -= min(45.0, (mean_slope - max_safe_slope) * 4.5)
    if unsafe_pct > 15.0:
        score -= min(35.0, (unsafe_pct - 15.0) * 1.2)
    if illum_pct < 65.0:
        score -= min(30.0, (65.0 - illum_pct) * 1.1)
    if is_psr:
        score -= 40.0
    if mean_tri > 0.45:
        score -= min(20.0, (mean_tri - 0.45) * 50.0)
    safety_score = max(0.0, round(score, 1))

    is_slope_safe = terrain.get("is_slope_safe", False)
    is_illum_safe = solar.get("is_illumination_safe", False)
    is_approved = is_slope_safe and is_illum_safe and (safety_score >= 60.0)
    status = "APPROVED" if is_approved else "REJECTED_HAZARD"

    # Determine execution mode and generate mission brief
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    execution_mode = "Offline Deterministic + MCP"
    brief = ""

    if api_key and api_key != "your_google_ai_studio_api_key_here":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                timeout=12.0,
            )
            prompt = f"""You are the TerraLuna Autonomous Mission Flight Director.
Generate a concise, authoritative executive mission brief evaluating this landing zone based on real MCP tool telemetry:

MISSION CONTEXT:
- Target Body / Dataset: {dataset_id}
- Coordinates: Latitude {lat:.4f}°, Longitude {lon:.4f}°
- Search Footprint: {radius_m}m radius
- Final Status Determination: {status}
- Composite Safety Score: {safety_score}/100

DETERMINISTIC TOOL TELEMETRY:
1. Terrain Hazards:
   - Mean Slope: {mean_slope:.2f}° (Lander Limit: {max_safe_slope}°)
   - Max Slope: {terrain.get('max_slope_deg', 0):.2f}°
   - Unsafe Slope Area: {unsafe_pct:.1f}%
   - Terrain Ruggedness Index (TRI): {mean_tri:.3f}m
   - Slope Safe Flag: {is_slope_safe}

2. Solar Illumination:
   - Sunlit Area: {illum_pct:.1f}% (Minimum Limit: 65%)
   - Sun Elevation Angle: {solar.get('sun_elevation_deg', 12)}°
   - Power Viability: {solar.get('solar_power_viability', 'UNKNOWN')}
   - Permanently Shadowed Region (PSR) Risk: {is_psr}

3. Regolith Composition:
   - Classification: {regolith.get('regolith_classification', 'N/A')}
   - FeO Content: {regolith.get('mean_feo_wt_pct', 'N/A')}%
   - Hydration Index (2.8um): {regolith.get('mean_hydration_index', 'N/A')}
   - ISRU Resource Rating: {regolith.get('isru_resource_potential', 'N/A')}

4. Earth Analog (Lonar Crater Testbed) Correlation:
   - Overall Fidelity Match: {analog.get('overall_analog_match_pct', 'N/A')}%
   - Recommendation: {analog.get('recommendation', 'N/A')}

HAZARD FLAGS:
{chr(10).join(['- ' + h for h in hazard_flags]) if hazard_flags else 'None detected.'}

Format the brief cleanly with Markdown headers (Executive Summary, Geomorphological Analysis, Solar Power Viability, ISRU Assessment, Flight Recommendation). End with a definitive Go / No-Go sign-off.
"""
            response = llm.invoke(prompt)
            brief = str(response.content)
            execution_mode = "Gemini LLM + MCP"
        except Exception as e:
            # Graceful fallback to deterministic brief on LLM connection failure
            execution_mode = f"Offline Deterministic + MCP (Gemini fallback: {type(e).__name__})"

    if not brief:
        # Deterministic Fallback Brief
        brief = f"""### EXECUTIVE MISSION AUDIT BRIEF
**Verdict**: `{status}` | **Safety Score**: **{safety_score}/100** | **Engine**: `{execution_mode}`
**Target**: Latitude {lat:.4f}°, Longitude {lon:.4f}° (Radius: {radius_m}m)

#### 1. Geomorphological & Terrain Hazard Assessment
- **Mean Slope**: {mean_slope:.2f}° (Threshold: {max_safe_slope}°) — {'[COMPLIANT]' if is_slope_safe else '[HAZARD EXCEEDED]'}
- **Peak Slope**: {terrain.get('max_slope_deg', 0):.2f}° | **Unsafe Fraction**: {unsafe_pct:.1f}%
- **Surface Ruggedness (TRI)**: {mean_tri:.3f}m (Lander structural limit: {LANDER_CONSTRAINTS['max_tri_m']}m)
- **Topographic Context**: Elevation spans {terrain.get('elevation_min_m', 0):.1f}m to {terrain.get('elevation_max_m', 0):.1f}m.

#### 2. Solar Illumination & Array Power Budget
- **Sunlit Fraction**: **{illum_pct:.1f}%** at {solar.get('sun_elevation_deg', 12)}° sun elevation.
- **Power Viability**: `{solar.get('solar_power_viability', 'N/A')}` ({'Sufficient for nominal battery recharge' if is_illum_safe else 'Risk of terminal lander freeze-out'}).
- **Permanently Shadowed Region (PSR)**: {'DETECTED - Cryogenic hazard' if is_psr else 'Negative - Nominal sun horizon'}.

#### 3. Regolith Geochemistry & ISRU Potential
- **Classification**: {regolith.get('regolith_classification', 'N/A')}
- **Mineral Proxies**: FeO: {regolith.get('mean_feo_wt_pct', 'N/A')} wt% | TiO2: {regolith.get('mean_tio2_wt_pct', 'N/A')} wt% | 2.8µm Hydration: {regolith.get('mean_hydration_index', 'N/A')}
- **ISRU Rating**: `{regolith.get('isru_resource_potential', 'N/A')}`

#### 4. Earth Analog Testbed Benchmark (Lonar Basalt Crater)
- **Fidelity Match**: **{analog.get('overall_analog_match_pct', 'N/A')}%**
- **Guidance**: {analog.get('recommendation', 'N/A')}

---
### FLIGHT CONTROL DIRECTIVE: {'GO FOR POWERED DESCENT' if is_approved else 'NO-GO // HAZARD ABORT INITIATED'}
"""

    slope_metrics = SlopeMetrics(
        mean_slope_deg=terrain.get("mean_slope_deg", 0.0),
        max_slope_deg=terrain.get("max_slope_deg", 0.0),
        min_slope_deg=terrain.get("min_slope_deg", 0.0),
        std_slope_deg=terrain.get("std_slope_deg", 0.0),
        unsafe_slope_pct=terrain.get("unsafe_slope_pct", 0.0),
        max_safe_slope_deg=max_safe_slope,
        mean_tri_m=terrain.get("mean_tri_m", 0.0),
        max_tri_m=terrain.get("max_tri_m", 0.0),
        is_slope_safe=is_slope_safe,
    )

    solar_metrics = SolarMetrics(
        sun_elevation_deg=solar.get("sun_elevation_deg", 12.0),
        sun_azimuth_deg=solar.get("sun_azimuth_deg", 135.0),
        sunlit_pct=solar.get("sunlit_pct", 0.0),
        shadowed_pct=solar.get("shadowed_pct", 0.0),
        min_required_pct=65.0,
        is_illumination_safe=is_illum_safe,
        is_permanently_shadowed_region=is_psr,
        solar_power_viability=solar.get("solar_power_viability", "UNKNOWN"),
    )

    analog_comp = EarthAnalogComparison(
        overall_analog_match_pct=analog.get("overall_analog_match_pct", 0.0),
        slope_similarity_pct=analog.get("slope_similarity_pct", 0.0),
        roughness_similarity_pct=analog.get("roughness_similarity_pct", 0.0),
        mineral_affinity_pct=analog.get("mineral_affinity_pct", 0.0),
        earth_analog_site=analog.get("earth_analog_site", "Lonar Crater"),
        recommendation=analog.get("recommendation", ""),
    )

    report = LandingZoneReport(
        status=status,
        safety_score=safety_score,
        dataset_id=dataset_id,
        target_coordinates={"lat": lat, "lon": lon},
        search_radius_m=radius_m,
        slope_metrics=slope_metrics,
        solar_metrics=solar_metrics,
        regolith_summary=regolith,
        earth_analog_similarity=analog_comp,
        hazard_flags=hazard_flags,
        mesh_artifact_path=terrain.get("mesh_artifact_path"),
        illumination_artifact_path=solar.get("illumination_artifact_path"),
        tool_trace=[ToolTraceItem(**t) for t in tool_trace_raw],
        execution_mode=execution_mode,
        mission_brief=brief,
    )

    return {
        "status": status,
        "safety_score": safety_score,
        "execution_mode": execution_mode,
        "mission_brief": brief,
        "final_report": report.model_dump(),
    }


def route_bounds_check(state: AgentState) -> str:
    """Conditional router based on spatial bounds validation."""
    if state.get("is_valid_bounds", True):
        return "analyze_terrain"
    return "synthesize_report"


# ==========================================
# LangGraph Assembly
# ==========================================

def create_landing_audit_graph():
    """Builds and compiles the TerraLuna LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("validate_bounds", node_validate_bounds)
    workflow.add_node("analyze_terrain", node_analyze_terrain)
    workflow.add_node("check_illumination", node_check_illumination)
    workflow.add_node("analyze_regolith", node_analyze_regolith)
    workflow.add_node("compare_earth_analog", node_compare_earth_analog)
    workflow.add_node("synthesize_report", node_synthesize_report)

    # Set Entry Point
    workflow.set_entry_point("validate_bounds")

    # Add Conditional Edge from Bounds Validation
    workflow.add_conditional_edges(
        "validate_bounds",
        route_bounds_check,
        {
            "analyze_terrain": "analyze_terrain",
            "synthesize_report": "synthesize_report",
        },
    )

    # Linear Execution Edges
    workflow.add_edge("analyze_terrain", "check_illumination")
    workflow.add_edge("check_illumination", "analyze_regolith")
    workflow.add_edge("analyze_regolith", "compare_earth_analog")
    workflow.add_edge("compare_earth_analog", "synthesize_report")
    workflow.add_edge("synthesize_report", END)

    return workflow.compile()


# Global compiled app
agent_graph_app = create_landing_audit_graph()


def run_landing_audit(
    dataset_id: str = "lunar_south_pole",
    lat: float = -89.494,
    lon: float = -0.680,
    radius_m: int = 250,
    sun_elevation_deg: float = 12.0,
    sun_azimuth_deg: float = 135.0,
    max_safe_slope_deg: float = 10.0,
) -> LandingZoneReport:
    """
    Main entry point for autonomous landing site validation.
    Executes LangGraph state machine over MCP tools and returns a validated LandingZoneReport.
    """
    initial_state: AgentState = {
        "dataset_id": dataset_id,
        "lat": lat,
        "lon": lon,
        "radius_m": radius_m,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_azimuth_deg": sun_azimuth_deg,
        "max_safe_slope_deg": max_safe_slope_deg,
        "is_valid_bounds": True,
        "out_of_bounds_reason": None,
        "terrain_result": None,
        "solar_result": None,
        "regolith_result": None,
        "analog_result": None,
        "status": "PENDING",
        "safety_score": 0.0,
        "hazard_flags": [],
        "tool_trace": [],
        "execution_mode": "Offline Deterministic + MCP",
        "mission_brief": "",
        "final_report": None,
    }

    final_state = agent_graph_app.invoke(initial_state)
    report_dict = final_state.get("final_report", {})
    return LandingZoneReport(**report_dict)
