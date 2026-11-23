"""
TerraLuna - Automated Benchmark & Verification Test Suite
Tests deterministic landing site audits, MCP server resources/tools,
and LangGraph agent decision-making for safe, hazard, and out-of-bounds coordinates.
"""

import json
import pytest
from pathlib import Path

from agent.graph import run_landing_audit
from mcp_server.server import (
    DATASET_REGISTRY,
    LANDER_CONSTRAINTS,
    analyze_terrain_hazards,
    compare_lunar_earth_analog,
    extract_regolith_composition,
    get_lander_constraints,
    get_mission_catalog,
    landing_site_audit_prompt,
    simulate_solar_illumination,
)


class TestTerraLunaLandingValidation:
    """End-to-end verification of landing site validation benchmarks."""

    def test_safe_ridge_approved(self):
        """
        Scenario A: Safe Ridge Landing Zone (Chandrayaan-2 style).
        Must pass with APPROVED, slope < 10 deg, illumination >= 65%, and safety score >= 80.
        """
        report = run_landing_audit(
            dataset_id="lunar_south_pole",
            lat=-89.494,
            lon=-0.680,
            radius_m=200,
            sun_elevation_deg=12.0,
            sun_azimuth_deg=135.0,
            max_safe_slope_deg=10.0,
        )

        assert report.status == "APPROVED", f"Expected APPROVED, got {report.status}"
        assert report.safety_score >= 80.0, f"Expected safety score >= 80, got {report.safety_score}"
        assert report.slope_metrics is not None
        assert report.slope_metrics.is_slope_safe is True
        assert report.slope_metrics.mean_slope_deg < 10.0
        assert report.slope_metrics.unsafe_slope_pct < 15.0
        assert report.solar_metrics is not None
        assert report.solar_metrics.is_illumination_safe is True
        assert report.solar_metrics.sunlit_pct >= 65.0
        assert len(report.hazard_flags) == 0
        assert len(report.tool_trace) >= 4
        assert report.mesh_artifact_path is not None
        assert Path(report.mesh_artifact_path).exists()

    def test_steep_crater_rim_rejected(self):
        """
        Scenario B: Steep Impact Crater Rim Wall.
        Must fail with REJECTED_HAZARD due to slope exceeding lander tolerance (>10 deg)
        and high unsafe slope percentage.
        """
        report = run_landing_audit(
            dataset_id="lunar_south_pole",
            lat=-89.506,
            lon=0.680,
            radius_m=200,
            sun_elevation_deg=12.0,
            sun_azimuth_deg=135.0,
            max_safe_slope_deg=10.0,
        )

        assert report.status == "REJECTED_HAZARD", f"Expected REJECTED_HAZARD, got {report.status}"
        assert report.safety_score < 60.0
        assert report.slope_metrics is not None
        assert report.slope_metrics.is_slope_safe is False
        assert report.slope_metrics.mean_slope_deg > 10.0
        assert report.slope_metrics.unsafe_slope_pct > 15.0
        assert len(report.hazard_flags) > 0
        assert any("slope" in h.lower() for h in report.hazard_flags)

    def test_out_of_bounds_refusal(self):
        """
        Scenario C: Out-of-Bounds Coordinates Safety Refusal.
        Agent must intercept coordinates outside mapped raster bounds and refuse descent burn.
        """
        report = run_landing_audit(
            dataset_id="lunar_south_pole",
            lat=-80.0,
            lon=10.0,
            radius_m=200,
        )

        assert report.status == "OUT_OF_BOUNDS", f"Expected OUT_OF_BOUNDS, got {report.status}"
        assert report.safety_score == 0.0
        assert len(report.hazard_flags) > 0
        assert "OUT_OF_BOUNDS" in report.hazard_flags[0]
        # Should not have executed terrain analysis
        assert report.slope_metrics is None

    def test_earth_analog_lonar_crater(self):
        """
        Scenario D: Earth Analog Testbed (Lonar Basalt Impact Crater).
        Verifies execution over terrestrial basalt raster datasets.
        """
        report = run_landing_audit(
            dataset_id="lonar_analog",
            lat=19.9760,
            lon=76.5080,
            radius_m=200,
        )

        assert report.status in ["APPROVED", "REJECTED_HAZARD"]
        assert report.slope_metrics is not None
        assert report.regolith_summary is not None
        assert "Lonar" in report.regolith_summary.get("geological_classification", "")

    def test_mcp_server_primitives(self):
        """
        Scenario E: FastMCP Resources, Prompts, and Tools Direct API Verification.
        """
        # 1. Catalog resource
        catalog_json = get_mission_catalog()
        catalog = json.loads(catalog_json)
        assert "available_datasets" in catalog
        assert "lunar_south_pole" in catalog["available_datasets"]
        assert "lonar_analog" in catalog["available_datasets"]

        # 2. Constraints resource
        constraints_json = get_lander_constraints()
        constraints = json.loads(constraints_json)
        assert constraints["max_slope_deg"] == 10.0
        assert constraints["min_illumination_pct"] == 65.0

        # 3. Prompt template
        audit_prompt = landing_site_audit_prompt("Shackleton Rim", -89.5, 0.0)
        assert "Shackleton Rim" in audit_prompt
        assert "Latitude -89.5000°" in audit_prompt

        # 4. Terrain tool
        terrain = analyze_terrain_hazards("lunar_south_pole", -89.494, -0.680, 150)
        assert "mean_slope_deg" in terrain
        assert "mean_tri_m" in terrain

        # 5. Solar tool
        solar = simulate_solar_illumination("lunar_south_pole", -89.494, -0.680, 12.0, 135.0)
        assert "sunlit_pct" in solar
        assert "solar_power_viability" in solar

        # 6. Regolith tool
        regolith = extract_regolith_composition("lunar_south_pole", -89.494, -0.680, 150)
        assert "regolith_classification" in regolith

        # 7. Earth Analog comparison tool
        comparison = compare_lunar_earth_analog(-89.494, -0.680)
        assert "overall_analog_match_pct" in comparison
        assert 0.0 <= comparison["overall_analog_match_pct"] <= 100.0
