"""
Marketing transformation functions for adstock and saturation effects.

This module implements various transformation functions commonly used in
marketing mix modeling to capture carryover effects and diminishing returns.
"""

import numpy as np
from scipy import signal
from typing import Union, Optional


def geometric_adstock(x: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to capture carryover effects.
    
    The geometric adstock model assumes that the effect of advertising
    decays exponentially over time:
    
    adstock_t = x_t + decay_rate * adstock_{t-1}
    
    Parameters
    ----------
    x : np.ndarray
        Input array (e.g., advertising spend or impressions)
    decay_rate : float
        Decay/retention rate between 0 and 1. Higher values indicate
        longer-lasting effects. Typical range: 0.3-0.9
        
    Returns
    -------
    np.ndarray
        Adstocked array
        
    Examples
    --------
    >>> spend = np.array([100, 0, 0, 0, 0])
    >>> geometric_adstock(spend, 0.5)
    array([100.  ,  50.  ,  25.  ,  12.5 ,   6.25])
    """
    if not 0 <= decay_rate <= 1:
        raise ValueError("decay_rate must be between 0 and 1")
    
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay_rate * adstocked[t-1]
    
    return adstocked


def delayed_adstock(x: np.ndarray, decay_rate: float, 
                    peak_delay: int, normalize: bool = True) -> np.ndarray:
    """
    Apply delayed adstock transformation where effects peak after a delay.
    
    This is useful for media channels where awareness or consideration
    builds up before conversion (e.g., TV brand campaigns).
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    decay_rate : float
        Decay rate between 0 and 1
    peak_delay : int
        Number of periods until peak effect (e.g., 2-3 for TV)
    normalize : bool, default=True
        Whether to normalize the adstock weights to sum to 1
        
    Returns
    -------
    np.ndarray
        Delayed adstocked array
    """
    if not 0 <= decay_rate <= 1:
        raise ValueError("decay_rate must be between 0 and 1")
    if peak_delay < 0:
        raise ValueError("peak_delay must be non-negative")
    
    max_lag = len(x)
    weights = np.zeros(max_lag)
    
    for t in range(max_lag):
        if t < peak_delay:
            weights[t] = (t / peak_delay) ** 2
        else:
            weights[t] = decay_rate ** (t - peak_delay)
    
    if normalize:
        weights = weights / weights.sum()
    
    # Apply convolution
    adstocked = signal.convolve(x, weights, mode='same')
    
    return adstocked


def weibull_adstock(x: np.ndarray, shape: float, scale: float, 
                    max_lag: int = 13) -> np.ndarray:
    """
    Apply Weibull adstock transformation for flexible decay patterns.
    
    The Weibull distribution allows for more flexible decay patterns
    including delayed peak effects and faster/slower decay rates.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    shape : float
        Shape parameter (k). k < 1: decreasing rate, k = 1: constant rate,
        k > 1: increasing then decreasing rate
    scale : float
        Scale parameter (λ). Controls how quickly the effect decays
    max_lag : int, default=13
        Maximum number of lags to consider (e.g., 13 weeks for quarterly)
        
    Returns
    -------
    np.ndarray
        Weibull adstocked array
    """
    from scipy.stats import weibull_min
    
    lags = np.arange(0, max_lag)
    # Weibull PDF as weights
    weights = weibull_min.pdf(lags, c=shape, scale=scale)
    weights = weights / weights.sum()  # Normalize
    
    # Apply convolution
    adstocked = signal.convolve(x, weights, mode='same')
    
    return adstocked


def hill_transform(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """
    Apply Hill saturation transformation to model diminishing returns.
    
    The Hill equation (also called Hill-Langmuir equation) creates an
    S-shaped curve that captures saturation effects:
    
    f(x) = x^k / (x^k + alpha^k)
    
    Parameters
    ----------
    x : np.ndarray
        Input array (should be non-negative)
    alpha : float
        Half-saturation point (inflection point of the curve).
        This is the input level at which output reaches 50% of maximum.
    k : float
        Shape parameter controlling steepness. Higher k = steeper curve.
        Typical range: 0.5-5.0
        
    Returns
    -------
    np.ndarray
        Saturated array (values between 0 and 1)
        
    Notes
    -----
    - alpha should be set relative to the typical input scale
    - k controls curve steepness: k=1 is fairly linear, k>2 is S-shaped
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    
    x = np.asarray(x)
    if np.any(x < 0):
        raise ValueError("Input array must be non-negative")
    
    return x**k / (x**k + alpha**k)


def logistic_saturation(x: np.ndarray, midpoint: float, 
                       steepness: float) -> np.ndarray:
    """
    Apply logistic saturation transformation.
    
    Alternative to Hill transformation using logistic function:
    
    f(x) = 1 / (1 + exp(-steepness * (x - midpoint)))
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    midpoint : float
        Point of maximum growth rate (inflection point)
    steepness : float
        Controls curve steepness. Higher = steeper transition
        
    Returns
    -------
    np.ndarray
        Saturated array (values between 0 and 1)
    """
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))


def michaelis_menten(x: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """
    Apply Michaelis-Menten saturation (simplified Hill with k=1).
    
    f(x) = vmax * x / (km + x)
    
    Parameters
    ----------
    x : np.ndarray
        Input array (non-negative)
    vmax : float
        Maximum response value
    km : float
        Michaelis constant (input level at 50% vmax)
        
    Returns
    -------
    np.ndarray
        Saturated array
    """
    if km <= 0:
        raise ValueError("km must be positive")
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    
    x = np.asarray(x)
    if np.any(x < 0):
        raise ValueError("Input array must be non-negative")
    
    return vmax * x / (km + x)


def create_lagged_features(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Create lagged features for distributed lag models.
    
    Parameters
    ----------
    x : np.ndarray
        Input 1D array
    max_lag : int
        Maximum number of lags to create
        
    Returns
    -------
    np.ndarray
        2D array where each column is a lagged version
        Shape: (len(x), max_lag + 1)
    """
    n = len(x)
    lagged = np.zeros((n, max_lag + 1))
    
    for lag in range(max_lag + 1):
        if lag == 0:
            lagged[:, lag] = x
        else:
            lagged[lag:, lag] = x[:-lag]
    
    return lagged


def apply_transformations(spend: np.ndarray, 
                         adstock_params: dict,
                         saturation_params: dict) -> np.ndarray:
    """
    Apply both adstock and saturation transformations in sequence.
    
    This is the typical workflow for MMM: first apply adstock to capture
    carryover effects, then apply saturation to capture diminishing returns.
    
    Parameters
    ----------
    spend : np.ndarray
        Raw marketing spend or impressions
    adstock_params : dict
        Parameters for adstock transformation. Must include 'type' key
        and relevant parameters for that type.
        Example: {'type': 'geometric', 'decay_rate': 0.7}
    saturation_params : dict
        Parameters for saturation transformation. Must include 'type' key.
        Example: {'type': 'hill', 'alpha': 50000, 'k': 2}
        
    Returns
    -------
    np.ndarray
        Fully transformed array
        
    Examples
    --------
    >>> spend = np.array([100, 150, 200, 100, 50])
    >>> adstock_params = {'type': 'geometric', 'decay_rate': 0.5}
    >>> saturation_params = {'type': 'hill', 'alpha': 150, 'k': 2}
    >>> transformed = apply_transformations(spend, adstock_params, saturation_params)
    """
    # Step 1: Apply adstock
    adstock_type = adstock_params.get('type', 'geometric')
    
    if adstock_type == 'geometric':
        adstocked = geometric_adstock(spend, adstock_params['decay_rate'])
    elif adstock_type == 'delayed':
        adstocked = delayed_adstock(
            spend, 
            adstock_params['decay_rate'],
            adstock_params['peak_delay']
        )
    elif adstock_type == 'weibull':
        adstocked = weibull_adstock(
            spend,
            adstock_params['shape'],
            adstock_params['scale']
        )
    else:
        raise ValueError(f"Unknown adstock type: {adstock_type}")
    
    # Step 2: Apply saturation
    saturation_type = saturation_params.get('type', 'hill')
    
    if saturation_type == 'hill':
        transformed = hill_transform(
            adstocked,
            saturation_params['alpha'],
            saturation_params['k']
        )
    elif saturation_type == 'logistic':
        transformed = logistic_saturation(
            adstocked,
            saturation_params['midpoint'],
            saturation_params['steepness']
        )
    elif saturation_type == 'michaelis_menten':
        transformed = michaelis_menten(
            adstocked,
            saturation_params['vmax'],
            saturation_params['km']
        )
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")
    
    return transformed


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create sample data
    spend = np.concatenate([
        np.zeros(5),
        np.ones(5) * 100,
        np.zeros(10)
    ])
    
    # Apply geometric adstock
    adstocked = geometric_adstock(spend, decay_rate=0.7)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(spend, label='Original Spend', marker='o')
    axes[0].plot(adstocked, label='Adstocked (decay=0.7)', marker='s')
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Adstock Effect')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Saturation curve
    x_range = np.linspace(0, 200, 100)
    saturated = hill_transform(x_range, alpha=100, k=2)
    
    axes[1].plot(x_range, saturated)
    axes[1].set_xlabel('Adstocked Spend')
    axes[1].set_ylabel('Saturated Effect')
    axes[1].set_title('Hill Saturation Curve (α=100, k=2)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformation_example.png', dpi=150, bbox_inches='tight')
    print("Example visualization saved to transformation_example.png")
