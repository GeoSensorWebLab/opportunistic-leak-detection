"""Tests for multi-day campaign planning."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimization.campaign import (
    DayPlan,
    CampaignState,
    plan_next_day,
    close_day,
    campaign_summary,
)
from models.measurement import Measurement
from data.mock_data import get_leak_sources, get_baseline_path


@pytest.fixture
def sources():
    return get_leak_sources()


@pytest.fixture
def baseline_path():
    return get_baseline_path()


@pytest.fixture
def wind_params():
    return {
        "wind_speed": 3.0,
        "wind_direction_deg": 270.0,
        "stability_class": "D",
    }


@pytest.fixture
def campaign():
    return CampaignState()


class TestCampaignState:
    def test_initial_state(self, campaign):
        assert len(campaign.days) == 0
        assert campaign.current_belief is None

    def test_to_dict_empty(self, campaign):
        d = campaign.to_dict()
        assert d["num_days"] == 0
        assert d["has_belief"] is False


class TestPlanNextDay:
    def test_first_day_generates_recommendations(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )
        assert isinstance(day, DayPlan)
        assert day.day_index == 0
        assert day.starting_belief is not None
        assert day.entropy_start > 0
        assert campaign.current_belief is not None

    def test_day_has_recommendations(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )
        # Should have some recommendations (mock data has sources near path)
        assert isinstance(day.recommendations, list)


class TestCloseDay:
    def test_close_updates_campaign(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )

        measurements = [
            Measurement(
                x=0.0, y=0.0,
                concentration_ppm=5.0,
                detected=True,
                wind_speed=3.0,
                wind_direction_deg=270.0,
                stability_class="D",
            ),
        ]

        close_day(campaign, day, measurements, sources)

        assert len(campaign.days) == 1
        assert day.ending_belief is not None
        assert day.entropy_end >= 0
        assert campaign.current_belief is not None

    def test_multi_day_accumulates(self, campaign, sources, wind_params, baseline_path):
        for i in range(2):
            day = plan_next_day(
                campaign, sources, wind_params, baseline_path,
                resolution=20.0,
            )
            meas = [
                Measurement(
                    x=50.0 * i, y=0.0,
                    concentration_ppm=3.0 + i,
                    detected=(i == 0),
                    wind_speed=3.0,
                    wind_direction_deg=270.0,
                    stability_class="D",
                ),
            ]
            close_day(campaign, day, meas, sources)

        assert len(campaign.days) == 2
        # Second day should start from first day's ending belief
        assert campaign.days[1].day_index == 1


class TestCampaignSummary:
    def test_summary_empty(self, campaign):
        summary = campaign_summary(campaign)
        assert summary["total_days"] == 0
        assert summary["total_measurements"] == 0

    def test_summary_after_one_day(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )
        close_day(campaign, day, [], sources)

        summary = campaign_summary(campaign)
        assert summary["total_days"] == 1
        assert summary["total_measurements"] == 0
        assert len(summary["entropy_per_day"]) == 1


class TestCampaignSerialization:
    def test_to_dict_has_belief(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )
        close_day(campaign, day, [], sources)

        d = campaign.to_dict()
        assert d["has_belief"] is True
        assert "current_belief" in d
        assert "grid_x" in d
        assert "grid_y" in d
        assert len(d["days"]) == 1
        assert "starting_belief" in d["days"][0]
        assert "ending_belief" in d["days"][0]

    def test_roundtrip_preserves_belief(self, campaign, sources, wind_params, baseline_path):
        day = plan_next_day(
            campaign, sources, wind_params, baseline_path,
            resolution=20.0,
        )
        close_day(campaign, day, [], sources)

        d = campaign.to_dict()
        restored = CampaignState.from_dict(d)

        assert restored.current_belief is not None
        np.testing.assert_array_almost_equal(
            restored.current_belief, campaign.current_belief,
        )
        assert restored.grid_x is not None
        assert restored.grid_y is not None
        assert len(restored.days) == 1
        assert restored.days[0].entropy_start == pytest.approx(campaign.days[0].entropy_start)
        assert restored.days[0].entropy_end == pytest.approx(campaign.days[0].entropy_end)

    def test_roundtrip_empty(self, campaign):
        d = campaign.to_dict()
        restored = CampaignState.from_dict(d)
        assert len(restored.days) == 0
        assert restored.current_belief is None
