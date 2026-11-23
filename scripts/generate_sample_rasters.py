"""
TerraLuna - Synthetic-but-Physically-Realistic GeoTIFF Generator
Generates realistic georeferenced DEM and spectral raster datasets:
1. Chandrayaan-2 TMC-2 / NASA LRO LOLA composite DEM at Lunar South Pole
2. Chandrayaan-2 IIRS multi-band hydration & mineral proxy raster
3. Lonar Crater Earth Analog DEM (SRTM/Cartosat basalt analog)
4. Lonar Crater Sentinel-2 mineral/iron proxy bands
"""

import math
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LUNAR_DIR = DATA_DIR / "lunar"
EARTH_DIR = DATA_DIR / "earth_analog"


def create_directories():
    """Ensure data directories exist."""
    LUNAR_DIR.mkdir(parents=True, exist_ok=True)
    EARTH_DIR.mkdir(parents=True, exist_ok=True)


def generate_lunar_south_pole_dem(output_path: Path):
    """
    Chandrayaan-2 TMC-2 & LRO LOLA style DEM around Lunar South Pole.
    Center: lat -89.500, lon 0.000.
    200x200 grid, 5m/pixel resolution.
    Features:
    - Safe flat ridge around (-89.494, -0.680) with slope ~ 3-5 deg.
    - Steep impact crater around (-89.506, +0.680) with 28 deg rim walls and deep floor.
    - Micro-boulder surface roughness and regional south polar elevation tilt.
    """
    print(f"[GENERATING] Lunar South Pole DEM -> {output_path}")
    nx, ny = 200, 200
    pixel_size_m = 5.0
    center_lon = 0.0
    center_lat = -89.5

    # Lunar physical constants (mean radius = 1737.4 km)
    m_per_deg_lat = (math.pi / 180.0) * 1737400.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    x_res_deg = pixel_size_m / m_per_deg_lon
    y_res_deg = pixel_size_m / m_per_deg_lat

    west = center_lon - (nx / 2.0) * x_res_deg
    north = center_lat + (ny / 2.0) * y_res_deg
    transform = from_origin(west, north, x_res_deg, y_res_deg)

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_m = (x_idx - nx / 2.0) * pixel_size_m
    y_m = (ny / 2.0 - y_idx) * pixel_size_m  # positive y is North

    # 1. Base undulating polar terrain (-4000 m base)
    base_elevation = -4000.0 + 8.0 * np.sin(x_m / 200.0) * np.cos(y_m / 200.0)

    # 2. Steep Impact Crater centered at (+180m East, -180m South)
    crater_cx, crater_cy = 180.0, -180.0
    dist_to_crater = np.sqrt((x_m - crater_cx) ** 2 + (y_m - crater_cy) ** 2)
    crater_radius = 160.0
    # Rim uplift
    rim_uplift = 18.0 * np.exp(-((dist_to_crater - crater_radius) ** 2) / (2 * 35.0 ** 2))
    # Crater bowl: target ~28 deg slope on inner wall
    bowl = np.where(dist_to_crater <= crater_radius, -85.0 * (1.0 - (dist_to_crater / crater_radius) ** 1.8), 0.0)

    # 3. Safe Flat Ridge Landing Zone centered at (-180m West, +180m North)
    ridge_cx, ridge_cy = -180.0, 180.0
    dist_to_ridge = np.sqrt((x_m - ridge_cx) ** 2 + (y_m - ridge_cy) ** 2)
    ridge_plateau = 15.0 * np.exp(-(dist_to_ridge ** 2) / (2 * 140.0 ** 2))

    # 4. Micro-boulder and regolith texture
    np.random.seed(42)
    roughness = gaussian_filter(np.random.normal(0.0, 0.10, size=(ny, nx)), sigma=1.2) * 1.5

    elevation = (base_elevation + rim_uplift + bowl + ridge_plateau + roughness).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(elevation, 1)
        dst.set_band_description(1, "Elevation (meters relative to lunar datum)")

    print(f"[OK] Lunar DEM generated: {elevation.shape}, Range: [{elevation.min():.1f}m, {elevation.max():.1f}m]")


def generate_lunar_spectral_hydration(output_path: Path):
    """
    Chandrayaan-2 IIRS style multi-band raster proxy:
    - Band 1: FeO content (wt %)
    - Band 2: TiO2 content (wt %)
    - Band 3: 2.8-micron Hydroxyl (OH) / H2O hydration band depth index (0.00 - 0.25)
    """
    print(f"[GENERATING] Lunar IIRS Spectral Raster -> {output_path}")
    nx, ny = 200, 200
    pixel_size_m = 5.0
    center_lon = 0.0
    center_lat = -89.5

    m_per_deg_lat = (math.pi / 180.0) * 1737400.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    x_res_deg = pixel_size_m / m_per_deg_lon
    y_res_deg = pixel_size_m / m_per_deg_lat

    west = center_lon - (nx / 2.0) * x_res_deg
    north = center_lat + (ny / 2.0) * y_res_deg
    transform = from_origin(west, north, x_res_deg, y_res_deg)

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_m = (x_idx - nx / 2.0) * pixel_size_m
    y_m = (ny / 2.0 - y_idx) * pixel_size_m

    crater_cx, crater_cy = 180.0, -180.0
    dist_to_crater = np.sqrt((x_m - crater_cx) ** 2 + (y_m - crater_cy) ** 2)

    np.random.seed(101)
    # Band 1: FeO wt % (typical highland anorthosite is 4.0 - 6.5%, impact melt inside crater ~ 8-10%)
    feo = 5.1 + 0.6 * np.random.uniform(0.0, 1.0, size=(ny, nx))
    feo += np.where(dist_to_crater < 160.0, 3.8 * (1.0 - dist_to_crater / 160.0), 0.0)

    # Band 2: TiO2 wt % (typical highlands 0.3 - 0.9%)
    tio2 = 0.55 + 0.2 * np.random.uniform(0.0, 1.0, size=(ny, nx))
    tio2 += np.where(dist_to_crater < 160.0, 0.7 * (1.0 - dist_to_crater / 160.0), 0.0)

    # Band 3: 2.8-micron OH/H2O Hydration Index (High in cold crater floor ~ 0.12 - 0.22, low on ridge ~ 0.01 - 0.03)
    hydration = 0.022 + 0.005 * np.random.normal(0, 0.5, size=(ny, nx))
    crater_hydration = 0.17 * np.clip(1.0 - (dist_to_crater / 150.0) ** 2, 0.0, 1.0)
    hydration = np.clip(hydration + crater_hydration, 0.001, 0.28)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 3,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(feo.astype(np.float32), 1)
        dst.set_band_description(1, "FeO_Weight_Percent")
        dst.write(tio2.astype(np.float32), 2)
        dst.set_band_description(2, "TiO2_Weight_Percent")
        dst.write(hydration.astype(np.float32), 3)
        dst.set_band_description(3, "Hydration_Band_Depth_2800nm")

    print(f"[OK] Lunar Spectral raster generated: FeO ~ [{feo.min():.1f}%, {feo.max():.1f}%], Hydration ~ [{hydration.min():.3f}, {hydration.max():.3f}]")


def generate_lonar_crater_dem(output_path: Path):
    """
    SRTM / Cartosat style DEM around Lonar Impact Crater, Maharashtra, India.
    Center: lat 19.976, lon 76.508.
    200x200 grid, ~10m resolution.
    """
    print(f"[GENERATING] Lonar Crater DEM -> {output_path}")
    nx, ny = 200, 200
    pixel_size_m = 10.0
    center_lon = 76.508
    center_lat = 19.976

    m_per_deg_lat = (math.pi / 180.0) * 6371000.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    x_res_deg = pixel_size_m / m_per_deg_lon
    y_res_deg = pixel_size_m / m_per_deg_lat

    west = center_lon - (nx / 2.0) * x_res_deg
    north = center_lat + (ny / 2.0) * y_res_deg
    transform = from_origin(west, north, x_res_deg, y_res_deg)

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_m = (x_idx - nx / 2.0) * pixel_size_m
    y_m = (ny / 2.0 - y_idx) * pixel_size_m
    dist = np.sqrt(x_m ** 2 + y_m ** 2)

    # Deccan basalt plain at ~480m elevation
    base_plateau = 480.0 + 3.0 * np.sin(x_m / 300.0)

    # Rim crest at radius ~ 650m, height +30m
    rim_radius = 650.0
    rim_uplift = 30.0 * np.exp(-((dist - rim_radius) ** 2) / (2 * 110.0 ** 2))

    # Inner crater depression down to saline lake at ~450m
    depression = np.where(dist < rim_radius, -115.0 * (1.0 - (dist / rim_radius) ** 2.0), 0.0)

    # Flat lake bed inside radius < 300m at 450m elevation
    lake_floor = np.where(dist < 300.0, 450.0 - (base_plateau + depression), 0.0)

    # Terrain texture
    np.random.seed(77)
    noise = gaussian_filter(np.random.normal(0, 0.8, size=(ny, nx)), sigma=1.4)

    dem = (base_plateau + rim_uplift + depression + lake_floor + noise).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dem, 1)
        dst.set_band_description(1, "Elevation (meters WGS84)")

    print(f"[OK] Lonar Crater DEM generated: Range: [{dem.min():.1f}m, {dem.max():.1f}m]")


def generate_lonar_sentinel2_bands(output_path: Path):
    """
    Sentinel-2 mineral proxy bands for Lonar Basaltic analog:
    - Band 1: Deccan Basalt Iron Oxide index
    - Band 2: Clay / Alteration Silicates index
    - Band 3: Saline Lake Water / Moisture index
    """
    print(f"[GENERATING] Lonar Sentinel-2 Mineral Bands -> {output_path}")
    nx, ny = 200, 200
    pixel_size_m = 10.0
    center_lon = 76.508
    center_lat = 19.976

    m_per_deg_lat = (math.pi / 180.0) * 6371000.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    x_res_deg = pixel_size_m / m_per_deg_lon
    y_res_deg = pixel_size_m / m_per_deg_lat

    west = center_lon - (nx / 2.0) * x_res_deg
    north = center_lat + (ny / 2.0) * y_res_deg
    transform = from_origin(west, north, x_res_deg, y_res_deg)

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    x_m = (x_idx - nx / 2.0) * pixel_size_m
    y_m = (ny / 2.0 - y_idx) * pixel_size_m
    dist = np.sqrt(x_m ** 2 + y_m ** 2)

    np.random.seed(88)
    basalt_iron = 1.40 + 0.12 * np.random.uniform(0, 1, size=(ny, nx))
    rim_mask = (dist > 350.0) & (dist < 750.0)
    basalt_iron += np.where(rim_mask, 0.60, 0.0)

    clay_idx = 0.85 + 0.1 * np.random.uniform(0, 1, size=(ny, nx))
    shore_mask = (dist >= 280.0) & (dist <= 450.0)
    clay_idx += np.where(shore_mask, 0.70, 0.0)

    water_idx = 0.04 + 0.02 * np.random.uniform(0, 1, size=(ny, nx))
    water_idx += np.where(dist < 300.0, 0.88, 0.0)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 3,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(basalt_iron.astype(np.float32), 1)
        dst.set_band_description(1, "Basalt_Iron_Oxide_Index")
        dst.write(clay_idx.astype(np.float32), 2)
        dst.set_band_description(2, "Clay_Alteration_Index")
        dst.write(water_idx.astype(np.float32), 3)
        dst.set_band_description(3, "Saline_Lake_Water_Index")

    print(f"[OK] Lonar Sentinel-2 bands generated successfully.")


def generate_all_sample_rasters():
    """Generates all 4 scientific GeoTIFF rasters."""
    create_directories()
    generate_lunar_south_pole_dem(LUNAR_DIR / "ch2_tmc2_south_pole_dem.tif")
    generate_lunar_spectral_hydration(LUNAR_DIR / "ch2_iirs_hydration.tif")
    generate_lonar_crater_dem(EARTH_DIR / "lonar_crater_dem.tif")
    generate_lonar_sentinel2_bands(EARTH_DIR / "lonar_sentinel2_bands.tif")
    print("\n[SUCCESS] All synthetic-but-physically-realistic rasters ready.")


if __name__ == "__main__":
    generate_all_sample_rasters()
