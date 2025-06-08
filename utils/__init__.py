"""
Utility functions for economic simulations.

This package provides mathematical and economic utilities used throughout the
economic simulation model, including statistical functions, inequality metrics,
and various economic indicators.
"""

# Import core math utilities
from .math_utils import (
    sign_with_small_bias,
    gini_coefficient,
    theil_index,
    atkinson_index,
    add_differential_privacy_noise,
    euclidean_distance,
    hyperbolic_distance,
    softmax,
    clip_norm,
    moving_average
)

# Import economic metrics
from .metrics import (
    calculate_price_index,
    calculate_gdp,
    calculate_unemployment_rate,
    calculate_distribution_metrics,
    calculate_plan_fulfillment,
    calculate_producer_metrics,
    calculate_consumer_metrics,
)

# Define what gets imported with "from utils import *"
__all__ = [
    # Math utilities
    'sign_with_small_bias',
    'gini_coefficient',
    'theil_index',
    'atkinson_index',
    'add_differential_privacy_noise',
    'euclidean_distance',
    'hyperbolic_distance',
    'softmax',
    'clip_norm',
    'moving_average',
    
    # Economic metrics
    'calculate_price_index',
    'calculate_gdp',
    'calculate_unemployment_rate',
    'calculate_distribution_metrics',
    'calculate_plan_fulfillment',
    'calculate_producer_metrics',
    'calculate_consumer_metrics'
]