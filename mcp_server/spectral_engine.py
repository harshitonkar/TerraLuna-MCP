"""
TerraLuna - Spectral Regolith Composition & Earth Analog Engine
Performs mineral proxy extraction (FeO, TiO2, 2.8um OH/H2O hydration index)
and quantitative Earth Analog testbed comparison (Lonar Deccan Basalt vs Lunar target).
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


def _get_meter_per_degree(tif_path: str, center_lat: float) -> Tuple[float, float]:
    """Computes meters per degree for longitude and latitude."""
    path_lower = str(tif_path).lower()
    is_lunar = "lunar" in path_lower or abs(center_lat) > 80.0
    radius_m = 1737400.0 if is_lunar else 6371000.0
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = (math.pi / 180.0) * radius_m
    m_per_deg_lon = (math.pi / 180.0) * radius_m * max(0.001, math.cos(lat_rad))
    return m_per_deg_lat, m_per_deg_lon


def compute_regolith_indices(
    tif_path: str,
    center_lat: float,
    center_lon: float,
    radius_m: float = 250.0,
) -> Dict[str, Any]:
    """
    Extracts mineralogical and hydration statistics from multi-band spectral rasters:
    - Chandrayaan-2 IIRS: FeO (wt%), TiO2 (wt%), and 2.8-micron OH/H2O band depth.
    - Lonar Sentinel-2: Basalt Iron Oxide index, Clay alteration, Saline water index.
    Classifies the landing zone regolith and assesses ISRU water/volatile potential.
    """
    tif_path_obj = Path(tif_path)
    if not tif_path_obj.exists():
        raise FileNotFoundError(f"Spectral raster file not found: {tif_path}")

    with rasterio.open(tif_path_obj) as src:
        bounds = src.bounds
        if not (bounds.left <= center_lon <= bounds.right and bounds.bottom <= center_lat <= bounds.top):
            raise ValueError(
                f"Target coordinates ({center_lat}, {center_lon}) outside spectral raster bounds."
            )

        row_center, col_center = src.index(center_lon, center_lat)
        m_per_deg_lat, m_per_deg_lon = _get_meter_per_degree(tif_path, center_lat)
        pixel_size_m = ((abs(src.transform.a) * m_per_deg_lon) + (abs(src.transform.e) * m_per_deg_lat)) / 2.0
        r_pix = max(5, int(math.ceil(radius_m / max(1.0, pixel_size_m))))

        col_off = max(0, col_center - r_pix)
        row_off = max(0, row_center - r_pix)
        width = min(src.width - col_off, 2 * r_pix + 1)
        height = min(src.height - row_off, 2 * r_pix + 1)

        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        data = src.read(window=window).astype(np.float64)  # (bands, H, W)
        descriptions = [src.descriptions[i] or f"Band_{i+1}" for i in range(src.count)]

    _, ny, nx = data.shape
    cy, cx = (row_center - row_off, col_center - col_off)
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    dist_m = np.sqrt(((y_coords - cy) * pixel_size_m) ** 2 + ((x_coords - cx) * pixel_size_m) ** 2)
    roi_mask = dist_m <= radius_m
    if not np.any(roi_mask):
        roi_mask = np.ones((ny, nx), dtype=bool)

    is_lunar = "lunar" in str(tif_path).lower()

    if is_lunar:
        # Band 1: FeO (wt%), Band 2: TiO2 (wt%), Band 3: Hydration 2800nm
        feo_roi = data[0][roi_mask]
        tio2_roi = data[1][roi_mask]
        hydr_roi = data[2][roi_mask]

        mean_feo = float(np.mean(feo_roi))
        mean_tio2 = float(np.mean(tio2_roi))
        mean_hydr = float(np.mean(hydr_roi))
        max_hydr = float(np.max(hydr_roi))

        # Regolith classification
        if mean_hydr >= 0.10:
            regolith_type = "Hydrated Polar Cold-Trap Regolith (Elevated Volatiles)"
            isru_rating = "HIGH_PRIORITY_VOLATILES"
        elif mean_feo >= 10.0:
            regolith_type = "Mare Basaltic Regolith (High FeO/TiO2)"
            isru_rating = "ILMENITE_RICH_ORE"
        elif mean_feo < 7.0 and mean_tio2 < 1.2:
            regolith_type = "Lunar Highland Anorthosite (Plagioclase-dominated, Low FeO)"
            isru_rating = "ALUMINUM_SILICON_FEEDSTOCK"
        else:
            regolith_type = "Intermediate Polar Regolith / Impact Melt Sheet"
            isru_rating = "MODERATE"

        return {
            "dataset_type": "Lunar_IIRS_Spectral",
            "mean_feo_wt_pct": round(mean_feo, 2),
            "mean_tio2_wt_pct": round(mean_tio2, 2),
            "mean_hydration_index": round(mean_hydr, 4),
            "max_hydration_index": round(max_hydr, 4),
            "regolith_classification": regolith_type,
            "isru_resource_potential": isru_rating,
            "is_hydration_anomaly_detected": bool(mean_hydr > 0.08 or max_hydr > 0.12),
        }
    else:
        # Earth Analog (Lonar Sentinel-2 bands)
        b1_roi = data[0][roi_mask]
        b2_roi = data[1][roi_mask]
        b3_roi = data[2][roi_mask]

        mean_fe = float(np.mean(b1_roi))
        mean_clay = float(np.mean(b2_roi))
        mean_water = float(np.mean(b3_roi))

        if mean_water > 0.5:
            classification = "Lonar Hypersaline Alkaline Crater Lake"
        elif mean_fe > 1.6:
            classification = "Shocked Deccan Basalt Rim Wall (Prime Terrestrial Lunar Analog)"
        else:
            classification = "Weathered Basalt Plain / Ejecta Blanket"

        return {
            "dataset_type": "Earth_Analog_Sentinel2",
            "mean_basalt_iron_index": round(mean_fe, 3),
            "mean_clay_alteration_index": round(mean_clay, 3),
            "mean_water_salinity_index": round(mean_water, 3),
            "geological_classification": classification,
            "terrestrial_analog_suitability": "EXCELLENT_BASALTIC_TESTBED" if mean_fe > 1.4 else "MODERATE",
        }


def compare_lunar_earth_analog(
    lunar_terrain_metrics: Dict[str, Any],
    lunar_spectral_metrics: Dict[str, Any],
    earth_terrain_metrics: Dict[str, Any],
    earth_spectral_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes multi-parametric similarity between target Lunar landing site and
    Lonar Crater Earth Analog testbed:
    - Topographic roughness & slope distribution similarity
    - Basaltic/iron mineralogical affinity
    - Lander testing fidelity score (0 - 100%)
    """
    # 1. Slope & Roughness match:
    l_slope = lunar_terrain_metrics.get("mean_slope_deg", 5.0)
    e_slope = earth_terrain_metrics.get("mean_slope_deg", 5.0)
    slope_diff = abs(l_slope - e_slope)
    slope_similarity = max(0.0, 100.0 - slope_diff * 6.0)

    l_tri = lunar_terrain_metrics.get("mean_tri_m", 0.3)
    e_tri = earth_terrain_metrics.get("mean_tri_m", 0.3)
    tri_diff = abs(l_tri - e_tri)
    tri_similarity = max(0.0, 100.0 - tri_diff * 80.0)

    # 2. Mineral / Basaltic affinity:
    # Lunar FeO weight percent (4% - 15%) correlates with Lonar Deccan basalt iron index (1.2 - 2.2)
    l_feo = lunar_spectral_metrics.get("mean_feo_wt_pct", 6.0)
    e_fe = earth_spectral_metrics.get("mean_basalt_iron_index", 1.5)
    # Scaled correlation
    feo_norm = (l_feo - 4.0) / 12.0  # 0 to 1
    fe_norm = (e_fe - 1.2) / 1.0     # 0 to 1
    mineral_similarity = max(0.0, 100.0 - abs(feo_norm - fe_norm) * 50.0)

    # Overall analog fidelity score (weighted composite)
    overall_match_pct = round(
        0.45 * slope_similarity + 0.25 * tri_similarity + 0.30 * mineral_similarity, 1
    )

    if overall_match_pct >= 75.0:
        recommendation = "EXCELLENT_ANALOG: Lonar basalt terrain accurately replicates landing site roughness and mechanical regolith friction."
    elif overall_match_pct >= 50.0:
        recommendation = "MODERATE_ANALOG: Suitable for rover traction and hazard avoidance sensor calibration."
    else:
        recommendation = "LOW_SIMILARITY: Topography or regolith composition diverges significantly from Earth analog."

    return {
        "overall_analog_match_pct": overall_match_pct,
        "slope_similarity_pct": round(slope_similarity, 1),
        "roughness_similarity_pct": round(tri_similarity, 1),
        "mineral_affinity_pct": round(mineral_similarity, 1),
        "earth_analog_site": "Lonar Impact Crater (Deccan Basalt, Maharashtra, India)",
        "recommendation": recommendation,
    }
