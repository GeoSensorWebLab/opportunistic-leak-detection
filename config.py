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
DETECTION_THRESHOLD_PPM = 5.0  # Minimum concentration for sensor to register (ppm)
METHANE_MOLAR_MASS = 16.04     # g/mol
AIR_DENSITY = 1.225            # kg/m^3 at sea level

# --- Tasking Optimizer ---
DEVIATION_EPSILON = 10.0       # Meters — prevents division-by-zero in cost function
MAX_DEVIATION_M = 200.0        # Max distance a worker should deviate from baseline path
TOP_K_RECOMMENDATIONS = 5      # Number of top waypoints to recommend

# --- Cache ---
CACHE_MAX_ENTRIES = 32         # Max entries for opportunity map cache

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
