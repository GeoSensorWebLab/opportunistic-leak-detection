"""Tests for Weather API abstraction."""

import sys
import os
import pytest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.weather import WindObservation, WeatherProvider, StubWeatherProvider


class TestWindObservation:
    def test_basic_creation(self):
        obs = WindObservation(speed=3.0, direction=270.0, stability_class="D")
        assert obs.speed == 3.0
        assert obs.direction == 270.0
        assert obs.stability_class == "D"

    def test_optional_fields_default_none(self):
        obs = WindObservation(speed=3.0, direction=270.0, stability_class="D")
        assert obs.timestamp is None
        assert obs.station_id is None

    def test_stability_class_uppercased(self):
        obs = WindObservation(speed=3.0, direction=270.0, stability_class="d")
        assert obs.stability_class == "D"

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError, match="speed"):
            WindObservation(speed=-1.0, direction=270.0, stability_class="D")

    def test_invalid_stability_raises(self):
        with pytest.raises(ValueError, match="stability class"):
            WindObservation(speed=3.0, direction=270.0, stability_class="Z")

    def test_all_stability_classes_valid(self):
        for cls in "ABCDEF":
            obs = WindObservation(speed=1.0, direction=0.0, stability_class=cls)
            assert obs.stability_class == cls


class TestStubWeatherProvider:
    def test_is_weather_provider(self):
        provider = StubWeatherProvider()
        assert isinstance(provider, WeatherProvider)

    def test_default_current_wind(self):
        provider = StubWeatherProvider()
        obs = provider.get_current_wind()
        assert obs.speed == 3.0
        assert obs.direction == 270.0
        assert obs.stability_class == "D"
        assert obs.station_id == "STUB-001"
        assert obs.timestamp is not None

    def test_custom_config(self):
        provider = StubWeatherProvider(
            speed=5.0, direction=180.0, stability_class="B", station_id="TEST-99"
        )
        obs = provider.get_current_wind()
        assert obs.speed == 5.0
        assert obs.direction == 180.0
        assert obs.stability_class == "B"
        assert obs.station_id == "TEST-99"

    def test_forecast_length(self):
        provider = StubWeatherProvider()
        forecast = provider.get_forecast(hours_ahead=4)
        assert len(forecast) == 4

    def test_forecast_default_length(self):
        provider = StubWeatherProvider()
        forecast = provider.get_forecast()
        assert len(forecast) == 6

    def test_forecast_entries_are_wind_observations(self):
        provider = StubWeatherProvider()
        for obs in provider.get_forecast(hours_ahead=3):
            assert isinstance(obs, WindObservation)
            assert obs.speed == 3.0
            assert obs.stability_class == "D"

    def test_forecast_has_timestamps(self):
        provider = StubWeatherProvider()
        forecast = provider.get_forecast(hours_ahead=3)
        for obs in forecast:
            assert obs.timestamp is not None
            assert isinstance(obs.timestamp, datetime)
