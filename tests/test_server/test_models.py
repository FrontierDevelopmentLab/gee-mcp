"""Tests for Pydantic models in ``gee_mcp.server.models``."""

import importlib

import pytest
from pydantic import ValidationError

from gee_mcp.config import SERVER_MODULE as PKG


@pytest.fixture()
def models_mod():
    """Return the imported models module."""
    return importlib.import_module(f"{PKG}.models")


class TestRegionParams:
    """Test ``RegionParams`` validation."""

    @staticmethod
    def test_accepts_lat_lon(models_mod):
        """Latitude + longitude is accepted."""
        params = models_mod.RegionParams(latitude=10.0, longitude=20.0)
        assert params.latitude == 10.0
        assert params.longitude == 20.0

    @staticmethod
    def test_accepts_bounding_box(models_mod):
        """Four-element bounding box is accepted."""
        params = models_mod.RegionParams(bounding_box=[0, 0, 1, 1])
        assert params.bounding_box == [0, 0, 1, 1]

    @staticmethod
    def test_rejects_no_location(models_mod):
        """Validation fails when no location is supplied."""
        with pytest.raises(ValidationError):
            models_mod.RegionParams()

    @staticmethod
    def test_rejects_multiple_locations(models_mod):
        """Validation fails when more than one location form is given."""
        with pytest.raises(ValidationError):
            models_mod.RegionParams(
                latitude=1.0, longitude=2.0, bounding_box=[0, 0, 1, 1]
            )

    @staticmethod
    def test_rejects_short_bounding_box(models_mod):
        """Validation fails when bounding box has wrong arity."""
        with pytest.raises(ValidationError):
            models_mod.RegionParams(bounding_box=[0, 0, 1])


class TestComputeIndexParams:
    """Test ``ComputeIndexParams`` cross-field validation."""

    @staticmethod
    def test_requires_index_or_expression(models_mod):
        """Either index_name or expression must be provided."""
        with pytest.raises(ValidationError):
            models_mod.ComputeIndexParams(
                latitude=10.0,
                longitude=20.0,
                start_date="2024-01-01",
                end_date="2024-02-01",
            )

    @staticmethod
    def test_rejects_both_index_and_expression(models_mod):
        """Cannot supply both index_name and expression."""
        with pytest.raises(ValidationError):
            models_mod.ComputeIndexParams(
                latitude=10.0,
                longitude=20.0,
                start_date="2024-01-01",
                end_date="2024-02-01",
                index_name="NDVI",
                expression="(B8 - B4) / (B8 + B4)",
            )

    @staticmethod
    def test_unknown_index_rejected(models_mod):
        """Unknown index_name fails validation."""
        with pytest.raises(ValidationError):
            models_mod.ComputeIndexParams(
                latitude=10.0,
                longitude=20.0,
                start_date="2024-01-01",
                end_date="2024-02-01",
                index_name="NOT_REAL",
            )


class TestMultiPeriodParams:
    """Test ``MultiPeriodParams`` cross-field validation."""

    @staticmethod
    def test_requires_two_periods(models_mod):
        """Fewer than two periods is rejected."""
        with pytest.raises(ValidationError):
            models_mod.MultiPeriodParams(
                latitude=0.0,
                longitude=0.0,
                periods=[
                    models_mod.DateRange(
                        label="a",
                        start_date="2024-01-01",
                        end_date="2024-02-01",
                    )
                ],
            )

    @staticmethod
    def test_threshold_required_for_threshold_area(models_mod):
        """threshold_area analysis requires a threshold value."""
        with pytest.raises(ValidationError):
            models_mod.MultiPeriodParams(
                latitude=0.0,
                longitude=0.0,
                analysis="threshold_area",
                periods=[
                    models_mod.DateRange(
                        label="a",
                        start_date="2024-01-01",
                        end_date="2024-02-01",
                    ),
                    models_mod.DateRange(
                        label="b",
                        start_date="2024-03-01",
                        end_date="2024-04-01",
                    ),
                ],
            )
