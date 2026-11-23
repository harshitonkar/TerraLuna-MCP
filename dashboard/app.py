"""
TERRALUNA // Autonomous Lunar & Terrestrial Landing-Site Validation
3D Mission Control Digital Twin & Agent Telemetry Dashboard
Responsive Aerospace UI/UX for Desktop, Laptop, Tablet, and Mobile
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

# Page configuration - responsive layout
st.set_page_config(
    page_title="TERRALUNA // Mission Control",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom Aerospace Dark Theme CSS with Responsive Breakpoints (Desktop, Laptop, Tablet, Mobile)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Dark Atmosphere & Typography */
    html, body, .stApp {
        background: radial-gradient(ellipse at 50% 0%, #0d1527 0%, #060911 70%, #030508 100%) !important;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Headings */
    h1, h2, h3, h4, h5 {
        font-family: 'Orbitron', -apple-system, sans-serif !important;
        letter-spacing: 0.04em;
        color: #f8fafc;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #070a12;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00f2fe;
    }

    /* Header Container */
    .header-container {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 1.2rem;
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(1.3rem, 2.8vw, 2.2rem);
        font-weight: 900;
        letter-spacing: 0.06em;
        background: linear-gradient(90deg, #00f2fe 0%, #38bdf8 45%, #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }

    .sub-title {
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(0.72rem, 1.2vw, 0.85rem);
        letter-spacing: 0.08em;
        margin-top: 0.35rem;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
    }
    .pill-badge-cyan {
        background: rgba(0, 242, 254, 0.1);
        border-color: rgba(0, 242, 254, 0.3);
        color: #38bdf8;
    }
    .pill-badge-green {
        background: rgba(0, 255, 135, 0.1);
        border-color: rgba(0, 255, 135, 0.3);
        color: #00ff87;
    }

    /* Status Badges */
    .status-banner {
        border-radius: 8px;
        padding: 12px 18px;
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(0.88rem, 1.4vw, 1.05rem);
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 1rem;
        box-sizing: border-box;
        width: 100%;
        line-height: 1.3;
    }
    .status-approved {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.12) 0%, rgba(56, 189, 248, 0.15) 100%);
        border: 1px solid #00ff87;
        color: #00ff87;
        box-shadow: 0 0 20px rgba(0, 255, 135, 0.2);
    }
    .status-hazard {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.15) 100%);
        border: 1px solid #ef4444;
        color: #f87171;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }
    .status-oob {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid #eab308;
        color: #facc15;
        box-shadow: 0 0 20px rgba(234, 179, 8, 0.2);
    }
    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        background: rgba(0, 0, 0, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* KPI Grid - Fully Responsive */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    @media (min-width: 1200px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
    }
    @media (max-width: 520px) {
        .kpi-grid {
            grid-template-columns: 1fr;
            gap: 8px;
        }
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s ease, transform 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        border-color: rgba(0, 242, 254, 0.4);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
    }
    .kpi-metric {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(1.1rem, 1.8vw, 1.45rem);
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .kpi-footer {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* 3D Visualizer Container */
    .viz-container {
        background: rgba(11, 17, 30, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 1rem;
    }

    /* Trace List & Code Viewers */
    .trace-card {
        background: #090e18;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #00f2fe;
        border-radius: 4px 6px 6px 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
    }
    .trace-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }

    /* Sidebar Restyling */
    section[data-testid="stSidebar"] {
        background-color: #070b14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stButton button {
        min-height: 44px;
        border-radius: 6px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Markdown Mission Brief Polish */
    .mission-brief-box {
        background: rgba(13, 20, 36, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .mission-brief-box h3, .mission-brief-box h4 {
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        color: #38bdf8;
    }
    .mission-brief-box p, .mission-brief-box li {
        color: #cbd5e1;
    }
    .mission-brief-box strong {
        color: #f1f5f9;
    }

    /* Mobile Adaptations */
    @media (max-width: 768px) {
        .header-container {
            padding: 0.2rem 0 0.8rem 0;
        }
        .main-title {
            font-size: 1.35rem;
        }
        .status-banner {
            font-size: 0.82rem;
            padding: 10px 12px;
        }
        .viz-container {
            padding: 6px;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Sidebar Controls & Configuration
# ==========================================
with st.sidebar:
    st.markdown("### 🛰️ MISSION CONTROL")

    # Agent Mode Badge
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_gemini = bool(gemini_key and gemini_key != "your_google_ai_studio_api_key_here")

    if has_gemini:
        st.markdown(
            "<div class='pill-badge pill-badge-cyan' style='width:100%; justify-content:center; padding:6px 10px; margin-bottom:12px;'>"
            "🤖 ACTIVE: Gemini LLM + FastMCP</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='pill-badge pill-badge-green' style='width:100%; justify-content:center; padding:6px 10px; margin-bottom:12px;'>"
            "⚡ ACTIVE: Offline Deterministic + FastMCP</div>",
            unsafe_allow_html=True,
        )

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
        help="Select a landing target preset or enter custom coordinates.",
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
            "Dataset Target",
            options=list(DATASET_REGISTRY.keys()),
            format_func=lambda k: DATASET_REGISTRY[k]["name"],
        )
        default_lat = DATASET_REGISTRY[dataset_id]["nominal_center"]["lat"]
        default_lon = DATASET_REGISTRY[dataset_id]["nominal_center"]["lon"]

    # Responsive Coordinate Inputs
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

    st.markdown("#### ⚙️ Lander Constraints")

    with st.expander("Adjust Engineering Parameters", expanded=False):
        radius_m = st.slider("Search Radius (m)", min_value=50, max_value=400, value=200, step=25)
        sun_elevation = st.slider("Sun Elevation (°)", min_value=2.0, max_value=35.0, value=12.0, step=1.0)
        sun_azimuth = st.slider("Sun Azimuth (°)", min_value=0.0, max_value=360.0, value=135.0, step=15.0)
        max_safe_slope = st.slider("Max Slope Limit (°)", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    run_btn = st.button("🚀 Run Autonomous Agent Validation", type="primary", use_container_width=True)

    st.markdown(
        "<div style='font-size:0.68rem; color:#64748b; margin-top:16px; font-family: monospace; text-align:center;'>"
        "TERRALUNA v1.0.0 // FASTMCP ARCHITECTURE<br>ISRO CH-2 TMC-2 + NASA LRO LOLA</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# Session State Execution
# ==========================================
if "report" not in st.session_state:
    with st.spinner("Initializing planetary digital twin & running baseline audit..."):
        st.session_state.report = run_landing_audit(
            dataset_id=dataset_id,
            lat=target_lat,
            lon=target_lon,
            radius_m=radius_m if 'radius_m' in locals() else 200,
            sun_elevation_deg=sun_elevation if 'sun_elevation' in locals() else 12.0,
            sun_azimuth_deg=sun_azimuth if 'sun_azimuth' in locals() else 135.0,
            max_safe_slope_deg=max_safe_slope if 'max_safe_slope' in locals() else 10.0,
        )

if run_btn:
    with st.spinner("🤖 Autonomous AI Agent executing FastMCP tools & validating landing zone..."):
        st.session_state.report = run_landing_audit(
            dataset_id=dataset_id,
            lat=target_lat,
            lon=target_lon,
            radius_m=radius_m if 'radius_m' in locals() else 200,
            sun_elevation_deg=sun_elevation if 'sun_elevation' in locals() else 12.0,
            sun_azimuth_deg=sun_azimuth if 'sun_azimuth' in locals() else 135.0,
            max_safe_slope_deg=max_safe_slope if 'max_safe_slope' in locals() else 10.0,
        )

report = st.session_state.report


# ==========================================
# Header Section
# ==========================================
st.markdown(f"""
<div class='header-container'>
    <div class='main-title'>TERRALUNA // MISSION CONTROL</div>
    <div class='sub-title'>
        <span>Autonomous Planetary Landing-Site Validation</span>
        <span class='pill-badge pill-badge-cyan'>TARGET: {report.target_coordinates['lat']:.4f}°, {report.target_coordinates['lon']:.4f}°</span>
        <span class='pill-badge'>RADIUS: {report.search_radius_m}m</span>
        <span class='pill-badge pill-badge-green'>{report.execution_mode}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# Adaptive Multi-Device Layout
# On Desktop/Laptop: 2 balanced columns
# On Tablet/Mobile: Flow smoothly
# ==========================================
col_viz, col_telemetry = st.columns([1.18, 1.0], gap="large")

with col_viz:
    st.markdown("#### 🪐 3D Digital Twin Elevation Mesh")

    # Layer selection
    overlay_mode = st.radio(
        "Color Surface Mesh By:",
        options=["Elevation (m)", "Slope Hazard (°)", "Solar Shadow Mask"],
        horizontal=True,
        label_visibility="collapsed",
    )

    mesh_path = report.mesh_artifact_path
    illum_path = report.illumination_artifact_path

    if mesh_path and Path(mesh_path).exists():
        with open(mesh_path, "r", encoding="utf-8") as f:
            mesh_data = json.load(f)

        elev_grid = np.array(mesh_data["elevation_grid"])
        slope_grid = np.array(mesh_data["slope_grid"])
        ny, nx = elev_grid.shape
        pix_x = mesh_data.get("pixel_size_x_m", 5.0)
        pix_y = mesh_data.get("pixel_size_y_m", 5.0)

        x_meters = (np.arange(nx) - nx / 2.0) * pix_x
        y_meters = (np.arange(ny) - ny / 2.0) * pix_y

        if overlay_mode == "Slope Hazard (°)":
            color_surface = slope_grid
            colorscale = "Plasma"
            cbar_title = "Slope (°)"
        elif overlay_mode == "Solar Shadow Mask":
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

        fig = go.Figure(
            data=[
                go.Surface(
                    z=elev_grid,
                    x=x_meters,
                    y=y_meters,
                    surfacecolor=color_surface,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=dict(text=cbar_title, font=dict(family="Orbitron", size=10, color="#94a3b8")),
                        tickfont=dict(family="JetBrains Mono", size=9, color="#94a3b8"),
                        len=0.70,
                        thickness=12,
                        x=1.02,
                    ),
                    contours_z=dict(show=True, usecolormap=True, highlightcolor="#00f2fe", project_z=True),
                    lighting=dict(ambient=0.50, diffuse=0.8, roughness=0.35, specular=0.2),
                ),
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[float(elev_grid[ny // 2, nx // 2]) + 8.0],
                    mode="markers+text",
                    marker=dict(
                        size=8,
                        color="#00ff87" if report.status == "APPROVED" else "#ef4444",
                        symbol="diamond",
                        line=dict(color="#ffffff", width=2),
                    ),
                    text=["TARGET LZ"],
                    textposition="top center",
                    textfont=dict(family="Orbitron", size=11, color="#ffffff"),
                    name="Candidate Touchdown Site",
                ),
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Meters East", font=dict(family="Orbitron", size=9, color="#94a3b8")),
                    backgroundcolor="#070c18",
                    gridcolor="#1e293b",
                    tickfont=dict(family="JetBrains Mono", size=8, color="#64748b"),
                ),
                yaxis=dict(
                    title=dict(text="Meters North", font=dict(family="Orbitron", size=9, color="#94a3b8")),
                    backgroundcolor="#070c18",
                    gridcolor="#1e293b",
                    tickfont=dict(family="JetBrains Mono", size=8, color="#64748b"),
                ),
                zaxis=dict(
                    title=dict(text="Elevation (m)", font=dict(family="Orbitron", size=9, color="#94a3b8")),
                    backgroundcolor="#070c18",
                    gridcolor="#1e293b",
                    tickfont=dict(family="JetBrains Mono", size=8, color="#64748b"),
                ),
                camera=dict(eye=dict(x=-1.45, y=-1.45, z=0.90)),
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1.2, z=0.52),
            ),
            height=540,
        )

        # Plotly configuration: responsive=True, scrollZoom=False prevents mobile page scroll locking
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "responsive": True,
                "scrollZoom": False,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
    else:
        st.warning("Spatial mesh artifact not found. Execute a valid coordinate audit to generate 3D mesh.")


with col_telemetry:
    st.markdown("#### 📊 Agent Telemetry & Scorecard")

    # Status Banner
    if report.status == "APPROVED":
        st.markdown(
            f"<div class='status-banner status-approved'>"
            f"<span>✅ APPROVED // GO FOR POWERED DESCENT</span>"
            f"<span class='score-badge'>{report.safety_score:.1f}/100</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif report.status == "REJECTED_HAZARD":
        st.markdown(
            f"<div class='status-banner status-hazard'>"
            f"<span>⚠️ REJECTED // CRITICAL HAZARDS DETECTED</span>"
            f"<span class='score-badge'>{report.safety_score:.1f}/100</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-banner status-oob'>"
            f"<span>⛔ OUT OF BOUNDS // SPATIAL REFUSAL</span>"
            f"<span class='score-badge'>{report.safety_score:.1f}/100</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 4 KPI Cards in 2x2 responsive CSS grid
    slope_val = f"{report.slope_metrics.mean_slope_deg:.2f}°" if report.slope_metrics else "N/A"
    slope_status = "Safe (<10°)" if (report.slope_metrics and report.slope_metrics.is_slope_safe) else "Hazard (>10°)"
    slope_border = "#00ff87" if (report.slope_metrics and report.slope_metrics.is_slope_safe) else "#ef4444"

    sun_val = f"{report.solar_metrics.sunlit_pct:.1f}%" if report.solar_metrics else "N/A"
    sun_status = "Nominal (≥65%)" if (report.solar_metrics and report.solar_metrics.is_illumination_safe) else "Critical Shadow"
    sun_border = "#38bdf8" if (report.solar_metrics and report.solar_metrics.is_illumination_safe) else "#ef4444"

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
    <div class='kpi-grid'>
        <div class='kpi-card' style='border-left: 3px solid {slope_border};'>
            <div class='kpi-label'>Mean Slope</div>
            <div class='kpi-metric'>{slope_val}</div>
            <div class='kpi-footer'>{slope_status}</div>
        </div>
        <div class='kpi-card' style='border-left: 3px solid {sun_border};'>
            <div class='kpi-label'>Sunlit Area</div>
            <div class='kpi-metric'>{sun_val}</div>
            <div class='kpi-footer'>{sun_status}</div>
        </div>
        <div class='kpi-card' style='border-left: 3px solid #818cf8;'>
            <div class='kpi-label'>Regolith Type</div>
            <div class='kpi-metric' style='font-size: clamp(0.88rem, 1.3vw, 1.15rem);'>{reg_short}</div>
            <div class='kpi-footer'>ISRU: {report.regolith_summary.get('isru_resource_potential', 'N/A') if report.regolith_summary else 'N/A'}</div>
        </div>
        <div class='kpi-card' style='border-left: 3px solid #c084fc;'>
            <div class='kpi-label'>Earth Analog Match</div>
            <div class='kpi-metric'>{analog_val}</div>
            <div class='kpi-footer'>Lonar Basalt Testbed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Tabs for Executive Brief, Live MCP Tool Trace, and Terrain Breakdown
    tab_brief, tab_trace, tab_constraints = st.tabs([
        "📜 Mission Brief",
        f"📡 Live MCP Trace ({len(report.tool_trace)})",
        "⚙️ Constraints & Specs",
    ])

    with tab_brief:
        st.markdown(f"<div class='mission-brief-box'>{report.mission_brief}</div>", unsafe_allow_html=True)

    with tab_trace:
        st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>Sequential MCP Tool & Resource Execution Telemetry:</div>", unsafe_allow_html=True)
        for idx, trace in enumerate(report.tool_trace):
            dur_color = "#00ff87" if trace.duration_ms < 15.0 else ("#38bdf8" if trace.duration_ms < 60.0 else "#f59e0b")
            st.markdown(f"""
            <div class='trace-card'>
                <div class='trace-header'>
                    <span><strong>Step {idx+1}:</strong> {trace.tool_name}</span>
                    <span style='color:{dur_color}; font-weight:600;'>{trace.duration_ms:.1f} ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.popover(f"Inspect JSON: {trace.tool_name.split('//')[-1]}"):
                st.json({"inputs": trace.inputs, "output_summary": trace.output_summary})

    with tab_constraints:
        st.markdown("##### Lander Engineering Thresholds")
        st.json(LANDER_CONSTRAINTS)
        if report.hazard_flags:
            st.markdown("##### Detected Hazard Flags")
            for h in report.hazard_flags:
                st.error(f"⚠️ {h}")
        else:
            st.success("✅ Zero structural or illumination hazards detected within search footprint.")
