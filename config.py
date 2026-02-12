"""
Global configuration and constants for the Methane Leak Opportunistic Tasking System.
"""

# --- Grid / Site Configuration ---
GRID_SIZE_M = 1000          # Site grid extent in meters (1km x 1km)
GRID_RESOLUTION_M = 5       # Cell size in meters
RECEPTOR_HEIGHT_M = 1.5     # Height at which field worker carries sensor (meters)

# --- Plume Defaults ---
DEFAULT_EMISSION_RATE = 0.5    # kg/s  (moderate leak)
DEFAULT_WIND_SPEED = 3.0       # m/s
DEFAULT_WIND_DIRECTION = 270   # Meteorological convention: direction wind comes FROM (degrees)
DEFAULT_STABILITY_CLASS = "D"  # Neutral stability

# --- Detection ---
SENSOR_MDL_PPM = 1.0           # Minimum Detection Limit — hard floor below which P(detect) = 0
DETECTION_THRESHOLD_PPM = 5.0  # Sigmoid midpoint — concentration at which P(detect) = 50%
DETECTION_STEEPNESS = 1.0      # Sigmoid steepness parameter (higher = sharper transition)
METHANE_MOLAR_MASS = 16.04     # g/mol
AIR_MOLAR_MASS = 28.97         # g/mol (dry air average)
AIR_DENSITY = 1.225            # kg/m^3 at sea level

# --- Tasking Optimizer ---
DEVIATION_EPSILON = 10.0       # Meters — prevents division-by-zero in cost function
MAX_DEVIATION_M = 200.0        # Max distance a worker should deviate from baseline path
TOP_K_RECOMMENDATIONS = 5      # Number of top waypoints to recommend
MIN_WAYPOINT_SEPARATION_M = 50.0  # Minimum distance between recommended waypoints (NMS)
CLUSTER_FRACTION = 0.08        # Fraction of total path length for waypoint clustering

# --- Cache ---
CACHE_MAX_ENTRIES = 32         # Max entries for opportunity map cache

# --- Route / Walking ---
WALKING_SPEED_MPS = 1.2          # Typical field walking speed (~4.3 km/h)
PATH_ARROW_INTERVAL_M = 80       # Spacing for direction arrows along paths
DETOUR_TOLERANCE_M = 5.0         # Distance threshold for identifying detour segments

# --- Visualization ---
COMPASS_POSITION = (-400, 400)  # (x, y) data-coord position of compass rose on plots

# --- Prior Belief Model ---
# Equipment type base leak rates (annual probability of significant leak)
# Source: EPA GHGRP & API 2014 compressor/component studies
EQUIPMENT_LEAK_RATES = {
    "compressor": 0.12,        # Highest risk: rotating equipment, seals
    "valve": 0.08,             # Packing and stem leaks
    "pipeline_junction": 0.06, # Flange and connector leaks
    "storage_tank": 0.05,      # Thief hatches, PRVs
    "wellhead": 0.04,          # Casing, tubing connections
    "separator": 0.07,         # Vessel connections
    "default": 0.05,           # Fallback for unknown types
}

# Age factor: leak probability increases with equipment age
# P_age = 1 + AGE_FACTOR_SCALE * (age / AGE_REFERENCE_YEARS)^AGE_FACTOR_EXPONENT
AGE_REFERENCE_YEARS = 20.0    # Reference age for normalization
AGE_FACTOR_SCALE = 1.0        # Scale of age contribution
AGE_FACTOR_EXPONENT = 1.5     # Superlinear aging (accelerating degradation)

# Production rate factor: higher throughput = more mechanical stress
PRODUCTION_RATE_REFERENCE_MCFD = 3000.0  # Reference production for normalization
PRODUCTION_RATE_SCALE = 0.3   # Moderate influence of production rate

# Inspection recency factor: longer since inspection = higher uncertainty
INSPECTION_DECAY_DAYS = 90.0  # Half-life for inspection benefit (days)

# Spatial prior kernel: how far each source's prior influence extends
PRIOR_KERNEL_RADIUS_M = 100.0  # Gaussian kernel radius for spatial prior

# --- Bayesian Update ---
FALSE_ALARM_RATE = 0.01           # P(detection | no leak) for Bayes denominator

# --- Wind Ensemble ---
DEFAULT_ENSEMBLE_SCENARIOS = 8    # Number of wind scenarios for ensemble averaging
DEFAULT_WIND_SPREAD_DEG = 30.0    # Half-spread for directional fan (degrees)

# --- Pasquill-Gifford Stability Classes ---
# Coefficients for sigma_y and sigma_z: sigma = a * x^b
# x in meters, sigma in meters
# Source: Turner (1970), adapted for continuous point source
DISPERSION_COEFFICIENTS = {
    "A": {"sigma_y": (0.3658, 0.9031), "sigma_z": (0.192, 1.2044)},
    "B": {"sigma_y": (0.2751, 0.9031), "sigma_z": (0.156, 1.0857)},
    "C": {"sigma_y": (0.2090, 0.9031), "sigma_z": (0.116, 0.9865)},
    "D": {"sigma_y": (0.1471, 0.9031), "sigma_z": (0.079, 0.9031)},
    "E": {"sigma_y": (0.1046, 0.9031), "sigma_z": (0.063, 0.8314)},
    "F": {"sigma_y": (0.0722, 0.9031), "sigma_z": (0.053, 0.7540)},
}
