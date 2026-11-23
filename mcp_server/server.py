"""
TerraLuna - FastMCP Geospatial Digital Twin Server
Exposes deterministic terrain, solar illumination, and spectral regolith
validation tools for Chandrayaan-2, NASA LRO, and Earth Analog (Lonar) datasets.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP

from mcp_server.spectral_engine import (
    compare_lunar_earth_analog as compute_analog_comparison,
    compute_regolith_indices,
)
from mcp_server.terrain_engine import (
    compute_solar_illumination,
    compute_terrain_hazards,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LUNAR_DEM = DATA_DIR / "lunar" / "ch2_tmc2_south_pole_dem.tif"
LUNAR_SPECTRAL = DATA_DIR / "lunar" / "ch2_iirs_hydration.tif"
EARTH_DEM = DATA_DIR / "earth_analog" / "lonar_crater_dem.tif"
EARTH_SPECTRAL = DATA_DIR / "earth_analog" / "lonar_sentinel2_bands.tif"

DATASET_REGISTRY = {
    "lunar_south_pole": {
        "name": "Lunar South Pole (Chandrayaan-2 TMC-2 / NASA LRO LOLA & IIRS)",
        "dem_path": str(LUNAR_DEM),
        "spectral_path": str(LUNAR_SPECTRAL),
        "target_body": "Moon",
        "nominal_center": {"lat": -89.500, "lon": 0.000},
        "bounds": {"lat_min": -89.5165, "lat_max": -89.4835, "lon_min": -1.8895, "lon_max": 1.8895},
        "resolution_m": 5.0,
        "sensors": ["TMC-2 High-Resolution Stereo", "LOLA Laser Altimeter", "IIRS Hyperspectral 0.8-5.0um"],
    },
    "lonar_analog": {
        "name": "Lonar Meteorite Impact Crater (Deccan Basalt Analog, Maharashtra, India)",
        "dem_path": str(EARTH_DEM),
        "spectral_path": str(EARTH_SPECTRAL),
        "target_body": "Earth",
        "nominal_center": {"lat": 19.976, "lon": 76.508},
        "bounds": {"lat_min": 19.9670, "lat_max": 19.9850, "lon_min": 76.4984, "lon_max": 76.5176},
        "resolution_m": 10.0,
        "sensors": ["SRTM / Cartosat-1 DEM", "Sentinel-2 MSI Mineral Reflectance"],
    },
}

LANDER_CONSTRAINTS = {
    "mission_class": "ISRO Chandrayaan / NASA Artemis High-Precision Lander",
    "max_slope_deg": 10.0,
    "max_tri_m": 0.45,
    "min_illumination_pct": 65.0,
    "max_crater_depth_m": 2.5,
    "allowable_unsafe_pixel_pct": 15.0,
    "solar_survival_window_days": 14.0,
}

# Initialize FastMCP Server
mcp = FastMCP("TerraLuna-MCP")


# ==========================================
# MCP Resources
# ==========================================
@mcp.resource("twin://missions/catalog")
def get_mission_catalog() -> str:
    """
    Returns the JSON catalog of available Lunar and Earth Analog
    geospatial DEM and spectral raster datasets, coordinate bounding boxes, and sensor specs.
    """
    catalog = {
        "system": "TerraLuna Geospatial Digital Twin",
        "version": "1.0.0",
        "available_datasets": DATASET_REGISTRY,
    }
    return json.dumps(catalog, indent=2)


@mcp.resource("twin://constraints/lander-specs")
def get_lander_constraints() -> str:
    """
    Returns the structural and operational landing threshold limits for the lander,
    including maximum allowable slope, roughness index, and minimum solar power illumination.
    """
    return json.dumps(LANDER_CONSTRAINTS, indent=2)


# ==========================================
# MCP Prompts
# ==========================================
@mcp.prompt()
def landing_site_audit_prompt(site_name: str, lat: float, lon: float) -> str:
    """
    Reusable mission-control audit prompt template for evaluating candidate landing coordinates.
    """
    return f"""MISSION CONTROL DIRECTIVE // AUTONOMOUS LANDING ZONE AUDIT
Target Candidate Site: {site_name}
Target Coordinates: Latitude {lat:.4f}°, Longitude {lon:.4f}°

Instructions:
1. Validate spatial coordinates against available planetary raster bounds.
2. Query the terrain hazard engine to compute Horn's 2D slope and Terrain Ruggedness Index (TRI).
3. Query the solar illumination simulation engine to assess shadow occlusions and solar array survival.
4. Extract spectral regolith composition (FeO, TiO2, 2.8um OH/H2O hydration index) for ISRU potential.
5. Benchmark target slope and basalt profile against Lonar Crater Earth analog testbed.
6. Synthesize an authoritative GO/NO-GO landing recommendation with definitive engineering rationale.
"""


# ==========================================
# MCP Tools
# ==========================================
@mcp.tool()
def analyze_terrain_hazards(
    dataset_id: str = "lunar_south_pole",
    lat: float = -89.494,
    lon: float = -0.680,
    radius_m: int = 250,
) -> Dict[str, Any]:
    """
    Computes deterministic terrain hazards (Horn's slope in degrees, Riley TRI ruggedness,
    percentage of unsafe slopes, and 3D mesh artifact generation) around target coordinates.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    ds_info = DATASET_REGISTRY[dataset_id]
    dem_path = ds_info["dem_path"]

    result = compute_terrain_hazards(
        tif_path=dem_path,
        center_lat=lat,
        center_lon=lon,
        radius_m=float(radius_m),
        max_safe_slope_deg=LANDER_CONSTRAINTS["max_slope_deg"],
    )
    result["dataset_id"] = dataset_id
    result["dataset_name"] = ds_info["name"]
    return result


@mcp.tool()
def simulate_solar_illumination(
    dataset_id: str = "lunar_south_pole",
    lat: float = -89.494,
    lon: float = -0.680,
    sun_elevation_deg: float = 12.0,
    sun_azimuth_deg: float = 135.0,
) -> Dict[str, Any]:
    """
    Performs ray-cast horizon shadow simulation to determine sunlit percentage,
    topographic occlusions, and permanent shadow hazard (PSR) for solar lander power viability.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    ds_info = DATASET_REGISTRY[dataset_id]
    dem_path = ds_info["dem_path"]

    result = compute_solar_illumination(
        tif_path=dem_path,
        center_lat=lat,
        center_lon=lon,
        radius_m=250.0,
        sun_elevation_deg=float(sun_elevation_deg),
        sun_azimuth_deg=float(sun_azimuth_deg),
    )
    result["dataset_id"] = dataset_id
    return result


@mcp.tool()
def extract_regolith_composition(
    dataset_id: str = "lunar_south_pole",
    lat: float = -89.494,
    lon: float = -0.680,
    radius_m: int = 250,
) -> Dict[str, Any]:
    """
    Extracts mineralogical and hydration statistics from multi-band spectral rasters
    (FeO wt%, TiO2 wt%, 2.8um OH/H2O hydration index) and classifies the regolith type.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    ds_info = DATASET_REGISTRY[dataset_id]
    spectral_path = ds_info["spectral_path"]

    result = compute_regolith_indices(
        tif_path=spectral_path,
        center_lat=lat,
        center_lon=lon,
        radius_m=float(radius_m),
    )
    result["dataset_id"] = dataset_id
    return result


@mcp.tool()
def compare_lunar_earth_analog(
    lunar_lat: float = -89.494,
    lunar_lon: float = -0.680,
    earth_dataset_id: str = "lonar_analog",
) -> Dict[str, Any]:
    """
    Computes multi-parametric similarity score (0-100%) between lunar target slope/basalt profile
    and Lonar Basalt Impact Crater Earth analog testbed in Maharashtra, India.
    """
    # 1. Compute target Lunar metrics
    lunar_dem = DATASET_REGISTRY["lunar_south_pole"]["dem_path"]
    lunar_spectral = DATASET_REGISTRY["lunar_south_pole"]["spectral_path"]

    lunar_terrain = compute_terrain_hazards(lunar_dem, lunar_lat, lunar_lon, 200.0)
    lunar_regolith = compute_regolith_indices(lunar_spectral, lunar_lat, lunar_lon, 200.0)

    # 2. Compute Earth Analog metrics at Lonar Crater basalt rim
    earth_dem = DATASET_REGISTRY["lonar_analog"]["dem_path"]
    earth_spectral = DATASET_REGISTRY["lonar_analog"]["spectral_path"]
    earth_lat = DATASET_REGISTRY["lonar_analog"]["nominal_center"]["lat"]
    earth_lon = DATASET_REGISTRY["lonar_analog"]["nominal_center"]["lon"]

    earth_terrain = compute_terrain_hazards(earth_dem, earth_lat, earth_lon, 200.0)
    earth_regolith = compute_regolith_indices(earth_spectral, earth_lat, earth_lon, 200.0)

    # 3. Benchmark comparison
    comparison = compute_analog_comparison(
        lunar_terrain_metrics=lunar_terrain,
        lunar_spectral_metrics=lunar_regolith,
        earth_terrain_metrics=earth_terrain,
        earth_spectral_metrics=earth_regolith,
    )
    return comparison


if __name__ == "__main__":
    mcp.run()
