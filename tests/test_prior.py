"""Tests for the prior belief model."""

import numpy as np
import pytest
from models.prior import compute_source_prior, compute_all_priors, create_spatial_prior


class TestComputeSourcePrior:
    """Tests for single-source prior computation."""

    def test_returns_bounded_probability(self):
        """Prior should always be in [0, 1]."""
        source = {
            "equipment_type": "compressor",
            "age_years": 50,
            "production_rate_mcfd": 10000.0,
            "last_inspection_days": 365,
        }
        p = compute_source_prior(source)
        assert 0.0 <= p <= 1.0

    def test_older_equipment_higher_prior(self):
        """Older equipment should have higher prior probability."""
        young = {"equipment_type": "wellhead", "age_years": 2,
                 "production_rate_mcfd": 1000.0, "last_inspection_days": 30}
        old = {"equipment_type": "wellhead", "age_years": 35,
               "production_rate_mcfd": 1000.0, "last_inspection_days": 30}
        assert compute_source_prior(old) > compute_source_prior(young)

    def test_compressor_higher_than_wellhead(self):
        """Compressor should have higher base risk than wellhead."""
        compressor = {"equipment_type": "compressor", "age_years": 10,
                      "production_rate_mcfd": 1000.0, "last_inspection_days": 60}
        wellhead = {"equipment_type": "wellhead", "age_years": 10,
                    "production_rate_mcfd": 1000.0, "last_inspection_days": 60}
        assert compute_source_prior(compressor) > compute_source_prior(wellhead)

    def test_higher_production_higher_prior(self):
        """Higher production rate should increase prior."""
        low_prod = {"equipment_type": "valve", "age_years": 10,
                    "production_rate_mcfd": 500.0, "last_inspection_days": 60}
        high_prod = {"equipment_type": "valve", "age_years": 10,
                     "production_rate_mcfd": 8000.0, "last_inspection_days": 60}
        assert compute_source_prior(high_prod) > compute_source_prior(low_prod)

    def test_recent_inspection_lower_prior(self):
        """Recently inspected equipment should have lower prior."""
        recent = {"equipment_type": "valve", "age_years": 10,
                  "production_rate_mcfd": 1000.0, "last_inspection_days": 5}
        stale = {"equipment_type": "valve", "age_years": 10,
                 "production_rate_mcfd": 1000.0, "last_inspection_days": 300}
        assert compute_source_prior(recent) < compute_source_prior(stale)

    def test_missing_attributes_uses_defaults(self):
        """Missing optional attributes should not raise errors."""
        minimal = {"name": "Test"}
        p = compute_source_prior(minimal)
        assert 0.0 <= p <= 1.0

    def test_zero_age_gives_base_rate_factor(self):
        """Zero age should contribute factor of 1.0 (no increase)."""
        source = {"equipment_type": "wellhead", "age_years": 0,
                  "production_rate_mcfd": 0.0, "last_inspection_days": 0}
        p = compute_source_prior(source)
        # With age=0, production=0, inspection=0 days: factors are all ~1.0
        # So prior should be close to base rate (0.04 for wellhead)
        assert 0.03 < p < 0.06


class TestComputeAllPriors:
    """Tests for batch prior computation."""

    def test_returns_correct_count(self, mock_sources):
        """Should return one prior per source."""
        priors = compute_all_priors(mock_sources)
        assert len(priors) == len(mock_sources)

    def test_all_bounded(self, mock_sources):
        """All priors should be in [0, 1]."""
        priors = compute_all_priors(mock_sources)
        assert all(0.0 <= p <= 1.0 for p in priors)

    def test_variation_across_sources(self, mock_sources):
        """Different sources should have different priors (not all identical)."""
        priors = compute_all_priors(mock_sources)
        assert len(set(round(p, 4) for p in priors)) > 1


class TestCreateSpatialPrior:
    """Tests for spatial prior grid projection."""

    def test_output_shape_matches_grid(self, small_grid):
        """Output should match grid dimensions."""
        X, Y = small_grid
        sources = [{"x": 0.0, "y": 0.0}]
        priors = [0.5]
        spatial = create_spatial_prior(X, Y, sources, priors)
        assert spatial.shape == X.shape

    def test_bounded_zero_one(self, small_grid):
        """Spatial prior should be in [0, 1] everywhere."""
        X, Y = small_grid
        sources = [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 50.0}]
        priors = [0.8, 0.6]
        spatial = create_spatial_prior(X, Y, sources, priors)
        assert np.all(spatial >= 0.0)
        assert np.all(spatial <= 1.0)

    def test_peak_at_source_location(self, small_grid):
        """Highest prior should be near the source location."""
        X, Y = small_grid
        sources = [{"x": 0.0, "y": 0.0}]
        priors = [0.5]
        spatial = create_spatial_prior(X, Y, sources, priors)
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(spatial), spatial.shape)
        peak_x = X[peak_idx]
        peak_y = Y[peak_idx]
        assert abs(peak_x) < 15, "Peak should be near source x"
        assert abs(peak_y) < 15, "Peak should be near source y"

    def test_decays_with_distance(self, small_grid):
        """Prior should decrease away from source."""
        X, Y = small_grid
        sources = [{"x": 0.0, "y": 0.0}]
        priors = [0.5]
        spatial = create_spatial_prior(X, Y, sources, priors, kernel_radius=50.0)
        center = spatial.shape[0] // 2
        # Values at center should be higher than at edges
        assert spatial[center, center] > spatial[0, 0]

    def test_multiple_sources_combine(self, small_grid):
        """Two sources should produce higher combined prior than either alone."""
        X, Y = small_grid
        src1 = [{"x": -30.0, "y": 0.0}]
        src2 = [{"x": 30.0, "y": 0.0}]
        both = [{"x": -30.0, "y": 0.0}, {"x": 30.0, "y": 0.0}]

        sp1 = create_spatial_prior(X, Y, src1, [0.3])
        sp2 = create_spatial_prior(X, Y, src2, [0.3])
        sp_both = create_spatial_prior(X, Y, both, [0.3, 0.3])

        # Combined should be >= either individual (complementary probability)
        assert np.all(sp_both >= sp1 - 1e-10)
        assert np.all(sp_both >= sp2 - 1e-10)

    def test_zero_prior_gives_zero_spatial(self, small_grid):
        """Sources with zero prior should not contribute."""
        X, Y = small_grid
        sources = [{"x": 0.0, "y": 0.0}]
        spatial = create_spatial_prior(X, Y, sources, [0.0])
        assert np.allclose(spatial, 0.0)
