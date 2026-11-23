"""
TERRALUNA // Autonomous Lunar & Terrestrial Landing-Site Validation
3D Mission Control Digital Twin & Agent Telemetry Dashboard
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure sample rasters exist on cold start (e.g. on Streamlit Community Cloud)
from scripts.generate_sample_rasters import generate_all_sample_rasters

LUNAR_DEM_PATH = PROJECT_ROOT / "data" / "lunar" / "ch2_tmc2_south_pole_dem.tif"
LUNAR_SPEC_PATH = PROJECT_ROOT / "data" / "lunar" / "ch2_iirs_hydration.tif"
EARTH_DEM_PATH = PROJECT_ROOT / "data" / "earth_analog" / "lonar_crater_dem.tif"
EARTH_SPEC_PATH = PROJECT_ROOT / "data" / "earth_analog" / "lonar_sentinel2_bands.tif"

if not (LUNAR_DEM_PATH.exists() and LUNAR_SPEC_PATH.exists() and EARTH_DEM_PATH.exists() and EARTH_SPEC_PATH.exists()):
    generate_all_sample_rasters()

from agent.graph import run_landing_audit
from mcp_server.server import DATASET_REGISTRY, LANDER_CONSTRAINTS

load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="TERRALUNA // Autonomous Mission Control",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Aerospace Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600&display=swap');

    /* Global Dark Atmosphere */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d131f 0%, #07090e 100%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.05em;
    }

    .main-header {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: 0px;
        text-transform: uppercase;
    }

    .sub-header {
        color: #7d8b99;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }

    /* Metric Cards */
    .kpi-container {
        display: flex;
        gap: 12px;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        flex: 1;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(0, 242, 254, 0.6);
        transform: translateY(-2px);
    }
    .kpi-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8b9bb4;
        text-transform: uppercase;
        margin-bottom: 4px;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #5c6b7d;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge-approved {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.15) 0%, rgba(96, 239, 255, 0.2) 100%);
        border: 1px solid #00ff87;
        color: #00ff87;
        padding: 10px 18px;
        border-radius: 6px;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.25);
        margin-bottom: 1.2rem;
    }
    .status-badge-hazard {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 140, 0, 0.2) 100%);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 10px 18px;
        border-radius: 6px;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.3);
        margin-bottom: 1.2rem;
    }
    .status-badge-oob {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid #ffc107;
        color: #ffc107;
        padding: 10px 18px;
        border-radius: 6px;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0d14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Trace Table */
    .trace-item {
        background: #0f141f;
        border-left: 3px solid #00f2fe;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Sidebar Controls
# ==========================================
with st.sidebar:
    st.markdown("### 🛰️ MISSION CONTROL")

    # Agent Mode Badge
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_gemini = bool(gemini_key and gemini_key != "your_google_ai_studio_api_key_here")

    if has_gemini:
        st.success("🤖 **Active Mode:** Gemini 2.5 Flash + MCP")
    else:
        st.info("⚡ **Active Mode:** Offline Deterministic + MCP")

    st.markdown("---")
    st.markdown("#### 🎯 Target Preset Selector")

    preset = st.selectbox(
        "Mission Target Presets",
        options=[
            "1. Lunar South Pole — Safe Ridge (Chandrayaan-2)",
            "2. Lunar South Pole — Steep Crater Rim (Hazard Test)",
            "3. Earth Analog — Lonar Basalt Crater (India)",
            "4. Custom Coordinates",
        ],
        index=0,
    )

    # Coordinate mapping based on preset
    if "Safe Ridge" in preset:
        dataset_id = "lunar_south_pole"
        default_lat = -89.4940
        default_lon = -0.6800
    elif "Steep Crater Rim" in preset:
        dataset_id = "lunar_south_pole"
        default_lat = -89.5060
        default_lon = 0.6800
    elif "Lonar Basalt" in preset:
        dataset_id = "lonar_analog"
        default_lat = 19.9760
        default_lon = 76.5080
    else:
        dataset_id = st.selectbox(
            "Dataset",
            options=list(DATASET_REGISTRY.keys()),
            format_func=lambda k: DATASET_REGISTRY[k]["name"],
        )
        default_lat = DATASET_REGISTRY[dataset_id]["nominal_center"]["lat"]
        default_lon = DATASET_REGISTRY[dataset_id]["nominal_center"]["lon"]

    # Coordinate Inputs
    col_lat, col_lon = st.columns(2)
    with col_lat:
        target_lat = st.number_input(
            "Latitude (°)",
            value=float(default_lat),
            format="%.4f",
            step=0.001,
        )
    with col_lon:
        target_lon = st.number_input(
            "Longitude (°)",
            value=float(default_lon),
            format="%.4f",
            step=0.001,
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Engineering Parameters")

    radius_m = st.slider("Search Footprint Radius (m)", min_value=50, max_value=400, value=200, step=25)
    sun_elevation = st.slider("Sun Elevation Angle (°)", min_value=2.0, max_value=35.0, value=12.0, step=1.0)
    sun_azimuth = st.slider("Sun Azimuth (°)", min_value=0.0, max_value=360.0, value=135.0, step=15.0)
    max_safe_slope = st.slider("Max Lander Slope Tolerance (°)", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    st.markdown("---")
    run_btn = st.button("🚀 Run Autonomous Agent Validation", type="primary", use_container_width=True)

    st.markdown(
        "<div style='font-size:0.7rem; color:#5c6b7d; margin-top:20px; font-family: monospace;'>"
        "TERRALUNA v1.0.0 // MCP PROTOCOL v1.0<br>ISRO CH-2 TMC-2 + NASA LRO LOLA</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# Session State Management
# ==========================================
if "report" not in st.session_state:
    # Run default Safe Ridge on first load
    with st.spinner("Initializing planetary digital twin & running baseline audit..."):
        st.session_state.report = run_landing_audit(
            dataset_id=dataset_id,
            lat=target_lat,
            lon=target_lon,
            radius_m=radius_m,
            sun_elevation_deg=sun_elevation,
            sun_azimuth_deg=sun_azimuth,
            max_safe_slope_deg=max_safe_slope,
        )

if run_btn:
    with st.spinner("🤖 Autonomous AI Agent executing FastMCP tools & validating landing zone..."):
        st.session_state.report = run_landing_audit(
            dataset_id=dataset_id,
            lat=target_lat,
            lon=target_lon,
            radius_m=radius_m,
            sun_elevation_deg=sun_elevation,
            sun_azimuth_deg=sun_azimuth,
            max_safe_slope_deg=max_safe_slope,
        )

report = st.session_state.report


# ==========================================
# Main Dashboard UI
# ==========================================
st.markdown("<div class='main-header'>TERRALUNA // MISSION CONTROL</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-header'>Autonomous Lunar & Terrestrial Landing-Site Validation Engine • "
    f"Target: Lat {report.target_coordinates['lat']:.4f}°, Lon {report.target_coordinates['lon']:.4f}°</div>",
    unsafe_allow_html=True,
)

# Top KPI Row
col_left, col_right = st.columns([1.15, 1.0], gap="large")

with col_left:
    st.markdown("### 🪐 3D DIGITAL TWIN ELEVATION MESH")

    # Layer overlay selector
    overlay_mode = st.radio(
        "Color Surface Mesh By:",
        options=["Elevation (m)", "Slope Hazard (°)", "Solar Illumination Mask"],
        horizontal=True,
    )

    # Load 3D mesh artifact if available
    mesh_path = report.mesh_artifact_path
    illum_path = report.illumination_artifact_path

    if mesh_path and Path(mesh_path).exists():
        with open(mesh_path, "r", encoding="utf-8") as f:
            mesh_data = json.load(f)

        elev_grid = np.array(mesh_data["elevation_grid"])
        slope_grid = np.array(mesh_data["slope_grid"])
        roi_mask = np.array(mesh_data["roi_mask"])
        ny, nx = elev_grid.shape
        pix_x = mesh_data.get("pixel_size_x_m", 5.0)
        pix_y = mesh_data.get("pixel_size_y_m", 5.0)

        x_meters = (np.arange(nx) - nx / 2.0) * pix_x
        y_meters = (np.arange(ny) - ny / 2.0) * pix_y

        # Select surface coloring
        if overlay_mode == "Slope Hazard (°)":
            color_surface = slope_grid
            colorscale = "Plasma"
            cbar_title = "Slope (°)"
        elif overlay_mode == "Solar Illumination Mask":
            if illum_path and Path(illum_path).exists():
                with open(illum_path, "r", encoding="utf-8") as f:
                    illum_data = json.load(f)
                color_surface = np.array(illum_data["illumination_grid"])
            else:
                color_surface = np.ones_like(elev_grid)
            colorscale = [[0, "#080a10"], [0.49, "#1a2133"], [0.5, "#ffd166"], [1.0, "#ffffff"]]
            cbar_title = "Sunlit (1/0)"
        else:
            color_surface = elev_grid
            colorscale = "Viridis"
            cbar_title = "Elevation (m)"

        # 3D Plotly Surface
        fig = go.Figure(
            data=[
                go.Surface(
                    z=elev_grid,
                    x=x_meters,
                    y=y_meters,
                    surfacecolor=color_surface,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=dict(text=cbar_title, font=dict(family="Orbitron", size=11, color="#8b9bb4")),
                        tickfont=dict(family="JetBrains Mono", size=10, color="#8b9bb4"),
                        len=0.75,
                        thickness=14,
                    ),
                    contours_z=dict(show=True, usecolormap=True, highlightcolor="#00f2fe", project_z=True),
                    lighting=dict(ambient=0.45, diffuse=0.8, roughness=0.3, specular=0.2),
                ),
                # Landing target pin
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[float(elev_grid[ny // 2, nx // 2]) + 8.0],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        color="#00ff87" if report.status == "APPROVED" else "#ff4b4b",
                        symbol="diamond",
                        line=dict(color="#ffffff", width=2),
                    ),
                    text=["TARGET LZ"],
                    textposition="top center",
                    textfont=dict(family="Orbitron", size=12, color="#ffffff"),
                    name="Candidate Touchdown Site",
                ),
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=10),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Meters East", font=dict(family="Orbitron", size=10, color="#8b9bb4")),
                    backgroundcolor="#0d1117",
                    gridcolor="#21262d",
                    tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280"),
                ),
                yaxis=dict(
                    title=dict(text="Meters North", font=dict(family="Orbitron", size=10, color="#8b9bb4")),
                    backgroundcolor="#0d1117",
                    gridcolor="#21262d",
                    tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280"),
                ),
                zaxis=dict(
                    title=dict(text="Elevation (m)", font=dict(family="Orbitron", size=10, color="#8b9bb4")),
                    backgroundcolor="#0d1117",
                    gridcolor="#21262d",
                    tickfont=dict(family="JetBrains Mono", size=9, color="#6b7280"),
                ),
                camera=dict(eye=dict(x=-1.45, y=-1.45, z=0.95)),
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1.2, z=0.55),
            ),
            height=580,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Spatial mesh artifact not found. Execute a valid coordinate audit to generate 3D mesh.")


with col_right:
    st.markdown("### 📊 AGENT TELEMETRY & SCORECARD")

    # Status Badge
    if report.status == "APPROVED":
        st.markdown(
            f"<div class='status-badge-approved'>✅ APPROVED // GO FOR POWERED DESCENT "
            f"(Score: {report.safety_score:.1f}/100)</div>",
            unsafe_allow_html=True,
        )
    elif report.status == "REJECTED_HAZARD":
        st.markdown(
            f"<div class='status-badge-hazard'>⚠️ REJECTED // CRITICAL HAZARDS DETECTED "
            f"(Score: {report.safety_score:.1f}/100)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-badge-oob'>⛔ OUT OF BOUNDS // SPATIAL REFUSAL "
            f"(Score: {report.safety_score:.1f}/100)</div>",
            unsafe_allow_html=True,
        )

    # 4 KPI Cards in 2x2 grid
    slope_val = f"{report.slope_metrics.mean_slope_deg:.2f}°" if report.slope_metrics else "N/A"
    slope_status = "Safe (<10°)" if (report.slope_metrics and report.slope_metrics.is_slope_safe) else "Hazard (>10°)"

    sun_val = f"{report.solar_metrics.sunlit_pct:.1f}%" if report.solar_metrics else "N/A"
    sun_status = "Nominal (≥65%)" if (report.solar_metrics and report.solar_metrics.is_illumination_safe) else "Critical Shadow"

    reg_val = report.regolith_summary.get("regolith_classification", "N/A") if report.regolith_summary else "N/A"
    if "Anorthosite" in reg_val:
        reg_short = "Highland Anorthosite"
    elif "Hydrated" in reg_val:
        reg_short = "Hydrated Cold Trap"
    elif "Basalt" in reg_val:
        reg_short = "Basaltic Regolith"
    else:
        reg_short = reg_val[:22]

    analog_val = f"{report.earth_analog_similarity.overall_analog_match_pct:.1f}%" if report.earth_analog_similarity else "N/A"

    st.markdown(f"""
    <div class='kpi-container'>
        <div class='kpi-card'>
            <div class='kpi-title'>Mean Slope</div>
            <div class='kpi-value'>{slope_val}</div>
            <div class='kpi-subtitle'>{slope_status}</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-title'>Sunlit Area</div>
            <div class='kpi-value'>{sun_val}</div>
            <div class='kpi-subtitle'>{sun_status}</div>
        </div>
    </div>
    <div class='kpi-container'>
        <div class='kpi-card'>
            <div class='kpi-title'>Regolith Type</div>
            <div class='kpi-value' style='font-size:1.05rem;'>{reg_short}</div>
            <div class='kpi-subtitle'>ISRU: {report.regolith_summary.get('isru_resource_potential', 'N/A') if report.regolith_summary else 'N/A'}</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-title'>Earth Analog Match</div>
            <div class='kpi-value'>{analog_val}</div>
            <div class='kpi-subtitle'>Lonar Basalt Testbed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live MCP Tool Execution Trace
    with st.expander(f"📡 Live MCP Tool & Resource Trace ({len(report.tool_trace)} calls)", expanded=False):
        for idx, trace in enumerate(report.tool_trace):
            st.markdown(f"""
            <div class='trace-item'>
                <strong>Step {idx+1}: {trace.tool_name}</strong> &nbsp;•&nbsp;
                <span style='color:#00ff87;'>{trace.duration_ms:.1f} ms</span>
                <br><span style='color:#7d8b99;'>Timestamp: {trace.timestamp}</span>
            </div>
            """, unsafe_allow_html=True)
            with st.popover(f"Inspect JSON: {trace.tool_name.split('//')[-1]}"):
                st.json({"inputs": trace.inputs, "output_summary": trace.output_summary})

    # Executive Mission Brief
    st.markdown("#### 📜 Executive Mission Brief")
    st.markdown(report.mission_brief)
