"""
Marketing Econometrics Package

A practical implementation of marketing econometrics concepts including:
- Transformation functions (adstock, saturation)
- Marketing Mix Models (MMM)
- Budget optimization
"""

__version__ = "0.1.0"

from .transformations import (
    geometric_adstock,
    delayed_adstock,
    weibull_adstock,
    hill_transform,
    logistic_saturation,
    michaelis_menten,
    apply_transformations
)

from .models import (
    RidgeMMM,
    BayesianMMM,
    cross_validate_mmm
)

from .optimization import (
    optimize_budget_allocation,
    marginal_roi_allocation,
    scenario_analysis,
    calculate_response_curve,
    incremental_budget_impact
)

__all__ = [
    # Transformations
    'geometric_adstock',
    'delayed_adstock',
    'weibull_adstock',
    'hill_transform',
    'logistic_saturation',
    'michaelis_menten',
    'apply_transformations',
    # Models
    'RidgeMMM',
    'BayesianMMM',
    'cross_validate_mmm',
    # Optimization
    'optimize_budget_allocation',
    'marginal_roi_allocation',
    'scenario_analysis',
    'calculate_response_curve',
    'incremental_budget_impact'
]
