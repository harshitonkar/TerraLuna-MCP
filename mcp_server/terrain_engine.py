"""
TerraLuna - Deterministic Geospatial Terrain & Solar Illumination Engine
Performs Horn's slope calculation, Terrain Ruggedness Index (TRI),
and ray-cast solar horizon shadow analysis on DEM rasters.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_meter_per_degree(tif_path: str, center_lat: float) -> Tuple[float, float]:
    """
    Computes meters per degree for latitude and longitude.
    Lunar bodies (~1737.4 km radius) vs Earth (~6371 km radius).
    """
    path_lower = str(tif_path).lower()
    is_lunar = "lunar" in path_lower or abs(center_lat) > 80.0

    radius_m = 1737400.0 if is_lunar else 6371000.0
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = (math.pi / 180.0) * radius_m
    m_per_deg_lon = (math.pi / 180.0) * radius_m * max(0.0001, math.cos(lat_rad))
    return m_per_deg_lat, m_per_deg_lon


def _extract_roi_window(
    src: rasterio.io.DatasetReader,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    m_per_deg_lat: float,
    m_per_deg_lon: float,
) -> Tuple[np.ndarray, Window, float, float, Tuple[int, int]]:
    """
    Clips DEM window around center coordinates and computes local pixel scale in meters.
    """
    bounds = src.bounds
    if not (bounds.left <= center_lon <= bounds.right and bounds.bottom <= center_lat <= bounds.top):
        raise ValueError(
            f"Target coordinates (lat: {center_lat}, lon: {center_lon}) fall outside dataset spatial bounds "
            f"[lat: {bounds.bottom:.4f} to {bounds.top:.4f}, lon: {bounds.left:.4f} to {bounds.right:.4f}]."
        )

    row_center, col_center = src.index(center_lon, center_lat)

    # Pixel resolution in meters
    x_res_deg = abs(src.transform.a)
    y_res_deg = abs(src.transform.e)
    pixel_size_x_m = x_res_deg * m_per_deg_lon
    pixel_size_y_m = y_res_deg * m_per_deg_lat
    avg_pixel_m = (pixel_size_x_m + pixel_size_y_m) / 2.0

    r_pix_x = max(6, int(math.ceil(radius_m / max(0.5, pixel_size_x_m))))
    r_pix_y = max(6, int(math.ceil(radius_m / max(0.5, pixel_size_y_m))))
    r_pix = max(r_pix_x, r_pix_y)

    col_off = max(0, col_center - r_pix)
    row_off = max(0, row_center - r_pix)
    width = min(src.width - col_off, 2 * r_pix + 1)
    height = min(src.height - row_off, 2 * r_pix + 1)

    window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
    dem = src.read(1, window=window).astype(np.float64)

    center_in_win = (row_center - row_off, col_center - col_off)
    return dem, window, pixel_size_x_m, pixel_size_y_m, center_in_win


def compute_terrain_hazards(
    tif_path: str,
    center_lat: float,
    center_lon: float,
    radius_m: float = 250.0,
    max_safe_slope_deg: float = 10.0,
) -> Dict[str, Any]:
    """
    Deterministic terrain hazard calculation:
    - Extracts DEM window around target coordinates
    - Computes Horn's slope in degrees using 2D finite gradients
    - Calculates Riley et al. Terrain Ruggedness Index (TRI) in meters
    - Computes unsafe slope percentage and lander landing risk
    - Saves downsampled 3D elevation & slope grid JSON artifact for Streamlit/Plotly
    """
    tif_path_obj = Path(tif_path)
    if not tif_path_obj.exists():
        raise FileNotFoundError(f"DEM raster file not found: {tif_path}")

    with rasterio.open(tif_path_obj) as src:
        m_per_deg_lat, m_per_deg_lon = _get_meter_per_degree(tif_path, center_lat)
        dem, window, pix_x_m, pix_y_m, (cy, cx) = _extract_roi_window(
            src, center_lat, center_lon, radius_m, m_per_deg_lat, m_per_deg_lon
        )

    ny, nx = dem.shape
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    dist_m = np.sqrt(((y_coords - cy) * pix_y_m) ** 2 + ((x_coords - cx) * pix_x_m) ** 2)
    roi_mask = dist_m <= radius_m

    if not np.any(roi_mask):
        roi_mask = np.ones((ny, nx), dtype=bool)

    # 1. Slope Calculation (Horn's method / 2D central gradients with anisotropic pixel sizes)
    dz_dy = np.gradient(dem, pix_y_m, axis=0)
    dz_dx = np.gradient(dem, pix_x_m, axis=1)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # 2. Terrain Ruggedness Index (TRI - Riley et al.)
    pad_dem = np.pad(dem, 1, mode="edge")
    sq_diff_sum = np.zeros_like(dem)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = pad_dem[1 + di : 1 + di + ny, 1 + dj : 1 + dj + nx]
            sq_diff_sum += (neighbor - dem) ** 2
    tri = np.sqrt(sq_diff_sum / 8.0)

    # 3. Metrics within circular landing footprint
    elev_roi = dem[roi_mask]
    slope_roi = slope_deg[roi_mask]
    tri_roi = tri[roi_mask]

    mean_slope = float(np.mean(slope_roi))
    max_slope = float(np.max(slope_roi))
    min_slope = float(np.min(slope_roi))
    std_slope = float(np.std(slope_roi))

    unsafe_pixels = int(np.sum(slope_roi > max_safe_slope_deg))
    total_pixels = int(len(slope_roi))
    unsafe_pct = float((unsafe_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    mean_tri = float(np.mean(tri_roi))
    max_tri = float(np.max(tri_roi))

    # Lander safety assessment
    is_safe = (mean_slope <= max_safe_slope_deg) and (unsafe_pct < 15.0) and (max_slope <= max_safe_slope_deg * 2.2)

    hazard_reasons = []
    if mean_slope > max_safe_slope_deg:
        hazard_reasons.append(f"Mean slope ({mean_slope:.1f}°) exceeds lander tolerance ({max_safe_slope_deg}°)")
    if unsafe_pct >= 15.0:
        hazard_reasons.append(f"Unsafe terrain fraction ({unsafe_pct:.1f}%) exceeds allowable threshold (<15%)")
    if max_tri > 0.85:
        hazard_reasons.append(f"High local surface ruggedness (TRI max: {max_tri:.2f}m) indicates boulder/crater wall risk")

    # 4. Save 3D mesh artifact for UI rendering
    step = max(1, ny // 80)
    sub_dem = dem[::step, ::step]
    sub_slope = slope_deg[::step, ::step]
    sub_mask = roi_mask[::step, ::step]

    artifact_name = f"mesh_terrain_{abs(center_lat):.3f}_{abs(center_lon):.3f}.json"
    artifact_path = ARTIFACTS_DIR / artifact_name

    mesh_payload = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_m": radius_m,
        "grid_shape": [int(sub_dem.shape[0]), int(sub_dem.shape[1])],
        "pixel_size_x_m": pix_x_m * step,
        "pixel_size_y_m": pix_y_m * step,
        "elevation_grid": sub_dem.tolist(),
        "slope_grid": sub_slope.tolist(),
        "roi_mask": sub_mask.tolist(),
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(mesh_payload, f)

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_m": radius_m,
        "elevation_min_m": round(float(np.min(elev_roi)), 2),
        "elevation_max_m": round(float(np.max(elev_roi)), 2),
        "elevation_mean_m": round(float(np.mean(elev_roi)), 2),
        "mean_slope_deg": round(mean_slope, 2),
        "max_slope_deg": round(max_slope, 2),
        "min_slope_deg": round(min_slope, 2),
        "std_slope_deg": round(std_slope, 2),
        "max_safe_slope_deg": max_safe_slope_deg,
        "unsafe_slope_pct": round(unsafe_pct, 2),
        "mean_tri_m": round(mean_tri, 3),
        "max_tri_m": round(max_tri, 3),
        "is_slope_safe": is_safe,
        "hazard_reasons": hazard_reasons,
        "mesh_artifact_path": str(artifact_path),
    }


def compute_solar_illumination(
    tif_path: str,
    center_lat: float,
    center_lon: float,
    radius_m: float = 250.0,
    sun_elevation_deg: float = 12.0,
    sun_azimuth_deg: float = 135.0,
) -> Dict[str, Any]:
    """
    Ray-cast solar horizon shadow analysis:
    - Casts reverse ray from every pixel towards the solar vector
    - Tests for topographic occlusions (cast shadows) along terrain horizon
    - Checks surface normal vs sun vector for self-shadowing
    - Evaluates solar power viability and permanent-shadow hazard
    """
    tif_path_obj = Path(tif_path)
    if not tif_path_obj.exists():
        raise FileNotFoundError(f"DEM raster file not found: {tif_path}")

    with rasterio.open(tif_path_obj) as src:
        m_per_deg_lat, m_per_deg_lon = _get_meter_per_degree(tif_path, center_lat)
        dem, window, pix_x_m, pix_y_m, (cy, cx) = _extract_roi_window(
            src, center_lat, center_lon, radius_m * 1.5, m_per_deg_lat, m_per_deg_lon
        )

    ny, nx = dem.shape
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    dist_m = np.sqrt(((y_coords - cy) * pix_y_m) ** 2 + ((x_coords - cx) * pix_x_m) ** 2)
    roi_mask = dist_m <= radius_m

    az_rad = math.radians(sun_azimuth_deg)
    el_rad = math.radians(sun_elevation_deg)

    # Unit vector pointing towards the sun:
    sun_dx = math.sin(az_rad) * math.cos(el_rad)
    sun_dy = -math.cos(az_rad) * math.cos(el_rad)
    sun_dz = math.sin(el_rad)
    tan_elev = math.tan(el_rad)

    # 1. Slope normal check (Self-shadowing: Lambertian cosine i <= 0)
    dz_dy = np.gradient(dem, pix_y_m, axis=0)
    dz_dx = np.gradient(dem, pix_x_m, axis=1)
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    cos_inc = (-dz_dx * sun_dx - dz_dy * sun_dy + sun_dz) / norm
    is_self_lit = cos_inc > 0.0

    # 2. Horizon ray-casting for cast shadows
    avg_pix_m = (pix_x_m + pix_y_m) / 2.0
    step_size_pix = 1.0
    dir_x = sun_dx / math.sqrt(sun_dx**2 + sun_dy**2 + 1e-9)
    dir_y = sun_dy / math.sqrt(sun_dx**2 + sun_dy**2 + 1e-9)
    step_m = step_size_pix * avg_pix_m

    max_steps = min(60, int(radius_m * 1.8 / step_m))
    is_cast_shadowed = np.zeros((ny, nx), dtype=bool)

    for k in range(1, max_steps + 1):
        ray_x = np.clip(np.round(x_coords + k * dir_x * step_size_pix).astype(int), 0, nx - 1)
        ray_y = np.clip(np.round(y_coords + k * dir_y * step_size_pix).astype(int), 0, ny - 1)

        terrain_h = dem[ray_y, ray_x]
        ray_h = dem + (k * step_m) * tan_elev
        is_cast_shadowed |= (terrain_h > ray_h)

    illumination_mask = is_self_lit & (~is_cast_shadowed)

    roi_total = int(np.sum(roi_mask))
    sunlit_pixels = int(np.sum(illumination_mask & roi_mask))
    illumination_pct = float((sunlit_pixels / roi_total) * 100.0) if roi_total > 0 else 0.0
    shadow_pct = round(100.0 - illumination_pct, 2)

    min_required_pct = 65.0
    is_safe = illumination_pct >= min_required_pct
    is_psr = illumination_pct < 20.0

    artifact_name = f"mesh_illumination_{abs(center_lat):.3f}_{abs(center_lon):.3f}.json"
    artifact_path = ARTIFACTS_DIR / artifact_name

    step = max(1, ny // 80)
    sub_illum = illumination_mask[::step, ::step].astype(int)
    payload = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_azimuth_deg": sun_azimuth_deg,
        "sunlit_pct": round(illumination_pct, 2),
        "illumination_grid": sub_illum.tolist(),
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_azimuth_deg": sun_azimuth_deg,
        "sunlit_pct": round(illumination_pct, 2),
        "shadowed_pct": shadow_pct,
        "min_required_pct": min_required_pct,
        "is_illumination_safe": is_safe,
        "is_permanently_shadowed_region": is_psr,
        "solar_power_viability": "OPTIMAL" if illumination_pct >= 80.0 else ("ADEQUATE" if is_safe else "CRITICAL_SHADOW"),
        "illumination_artifact_path": str(artifact_path),
    }
