"""
Marketing budget optimization utilities.

This module provides functions for:
- Budget allocation optimization
- Response curve optimization
- Scenario planning
- What-if analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable, Optional
import warnings


def optimize_budget_allocation(
    response_functions: Dict[str, Callable],
    total_budget: float,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'SLSQP'
) -> Dict:
    """
    Optimize budget allocation across channels to maximize total response.
    
    Parameters
    ----------
    response_functions : dict
        Dictionary mapping channel names to response functions.
        Each function takes spend amount and returns response.
    total_budget : float
        Total budget to allocate across channels
    bounds : dict, optional
        Min/max spend per channel. Format: {'channel': (min, max)}
        If None, bounds are (0, total_budget) for each channel
    method : str, default='SLSQP'
        Optimization method: 'SLSQP' or 'differential_evolution'
        
    Returns
    -------
    dict
        Optimization results including optimal allocation and expected return
        
    Examples
    --------
    >>> def tv_response(x): return 1000 * np.sqrt(x)
    >>> def digital_response(x): return 800 * np.log1p(x)
    >>> response_functions = {'TV': tv_response, 'Digital': digital_response}
    >>> result = optimize_budget_allocation(response_functions, 100000)
    """
    channels = list(response_functions.keys())
    n_channels = len(channels)
    
    # Set bounds
    if bounds is None:
        bounds_list = [(0, total_budget) for _ in range(n_channels)]
    else:
        bounds_list = [bounds.get(ch, (0, total_budget)) for ch in channels]
    
    # Objective function (negative because we minimize)
    def objective(allocation):
        total_response = sum(
            response_functions[ch](allocation[i]) 
            for i, ch in enumerate(channels)
        )
        return -total_response  # Negative for maximization
    
    # Budget constraint
    def budget_constraint(allocation):
        return total_budget - np.sum(allocation)
    
    # Initial guess (equal allocation)
    x0 = np.array([total_budget / n_channels] * n_channels)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': budget_constraint}
    ]
    
    if method == 'SLSQP':
        # Sequential Least Squares Programming
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        optimal_allocation = result.x
        total_response = -result.fun
        
    elif method == 'differential_evolution':
        # Global optimization for non-convex problems
        # Adjust bounds for budget constraint
        result = differential_evolution(
            lambda x: objective(x) if abs(sum(x) - total_budget) < 1e-6 else 1e10,
            bounds_list,
            seed=42,
            maxiter=1000
        )
        
        optimal_allocation = result.x
        # Normalize to exact budget
        optimal_allocation = optimal_allocation * (total_budget / optimal_allocation.sum())
        total_response = -objective(optimal_allocation)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate individual responses
    individual_responses = {
        ch: response_functions[ch](optimal_allocation[i])
        for i, ch in enumerate(channels)
    }
    
    # Calculate current allocation ROI
    roi_by_channel = {
        ch: individual_responses[ch] / optimal_allocation[i] if optimal_allocation[i] > 0 else 0
        for i, ch in enumerate(channels)
    }
    
    return {
        'optimal_allocation': dict(zip(channels, optimal_allocation)),
        'total_response': total_response,
        'individual_responses': individual_responses,
        'roi_by_channel': roi_by_channel,
        'success': result.success if hasattr(result, 'success') else True
    }


def marginal_roi_allocation(
    response_functions: Dict[str, Callable],
    total_budget: float,
    step_size: float = 100,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict:
    """
    Allocate budget using marginal ROI principle.
    
    Iteratively allocates budget to the channel with highest marginal ROI
    until budget is exhausted.
    
    Parameters
    ----------
    response_functions : dict
        Channel response functions
    total_budget : float
        Total budget to allocate
    step_size : float, default=100
        Budget increment per iteration
    bounds : dict, optional
        Min/max constraints per channel
        
    Returns
    -------
    dict
        Allocation results
    """
    channels = list(response_functions.keys())
    
    # Initialize allocation
    allocation = {ch: 0.0 for ch in channels}
    
    # Set bounds
    if bounds is None:
        bounds = {ch: (0, total_budget) for ch in channels}
    
    # Initialize with minimum required spend
    for ch in channels:
        min_spend = bounds[ch][0]
        allocation[ch] = min_spend
    
    remaining_budget = total_budget - sum(allocation.values())
    
    if remaining_budget < 0:
        raise ValueError("Minimum bounds exceed total budget")
    
    # Track previous responses
    prev_responses = {ch: response_functions[ch](allocation[ch]) for ch in channels}
    
    # Iteratively allocate based on marginal ROI
    while remaining_budget >= step_size:
        marginal_rois = {}
        
        for ch in channels:
            # Check if we can add more to this channel
            if allocation[ch] + step_size <= bounds[ch][1]:
                # Calculate marginal response
                new_response = response_functions[ch](allocation[ch] + step_size)
                marginal_response = new_response - prev_responses[ch]
                marginal_roi = marginal_response / step_size
                marginal_rois[ch] = marginal_roi
            else:
                marginal_rois[ch] = -np.inf  # Can't allocate more
        
        # Allocate to channel with highest marginal ROI
        best_channel = max(marginal_rois, key=marginal_rois.get)
        
        if marginal_rois[best_channel] <= 0:
            # No positive marginal returns left
            break
        
        allocation[best_channel] += step_size
        prev_responses[best_channel] = response_functions[best_channel](allocation[best_channel])
        remaining_budget -= step_size
    
    # Calculate final metrics
    total_response = sum(
        response_functions[ch](allocation[ch]) for ch in channels
    )
    
    individual_responses = {
        ch: response_functions[ch](allocation[ch]) for ch in channels
    }
    
    roi_by_channel = {
        ch: individual_responses[ch] / allocation[ch] if allocation[ch] > 0 else 0
        for ch in channels
    }
    
    return {
        'optimal_allocation': allocation,
        'total_response': total_response,
        'individual_responses': individual_responses,
        'roi_by_channel': roi_by_channel,
        'unallocated_budget': remaining_budget
    }


def scenario_analysis(
    response_functions: Dict[str, Callable],
    scenarios: List[Dict[str, float]],
    scenario_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze multiple budget allocation scenarios.
    
    Parameters
    ----------
    response_functions : dict
        Channel response functions
    scenarios : list of dict
        List of allocation dictionaries to analyze
        Example: [{'TV': 50000, 'Digital': 30000}, ...]
    scenario_names : list, optional
        Names for each scenario
        
    Returns
    -------
    pd.DataFrame
        Comparison of scenarios
    """
    if scenario_names is None:
        scenario_names = [f"Scenario_{i+1}" for i in range(len(scenarios))]
    
    results = []
    
    for name, scenario in zip(scenario_names, scenarios):
        total_spend = sum(scenario.values())
        
        # Calculate responses
        responses = {
            ch: response_functions[ch](scenario.get(ch, 0))
            for ch in response_functions.keys()
        }
        
        total_response = sum(responses.values())
        overall_roi = total_response / total_spend if total_spend > 0 else 0
        
        result = {
            'scenario': name,
            'total_spend': total_spend,
            'total_response': total_response,
            'overall_roi': overall_roi
        }
        
        # Add channel-specific metrics
        for ch in response_functions.keys():
            result[f'{ch}_spend'] = scenario.get(ch, 0)
            result[f'{ch}_response'] = responses[ch]
            spend = scenario.get(ch, 0)
            result[f'{ch}_roi'] = responses[ch] / spend if spend > 0 else 0
        
        results.append(result)
    
    return pd.DataFrame(results)


def calculate_response_curve(
    response_function: Callable,
    spend_range: Tuple[float, float],
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate response curve for a channel.
    
    Parameters
    ----------
    response_function : callable
        Function mapping spend to response
    spend_range : tuple
        (min_spend, max_spend) to evaluate
    n_points : int, default=100
        Number of points to evaluate
        
    Returns
    -------
    spend_values : np.ndarray
        Spend levels evaluated
    response_values : np.ndarray
        Corresponding responses
    """
    spend_values = np.linspace(spend_range[0], spend_range[1], n_points)
    response_values = np.array([response_function(s) for s in spend_values])
    
    return spend_values, response_values


def find_optimal_spend_per_channel(
    response_function: Callable,
    target_roi: float,
    spend_range: Tuple[float, float] = (0, 1e6)
) -> float:
    """
    Find optimal spend level to achieve target ROI.
    
    Parameters
    ----------
    response_function : callable
        Channel response function
    target_roi : float
        Desired ROI (e.g., 2.0 for 2x return)
    spend_range : tuple, default=(0, 1e6)
        Search range for spend
        
    Returns
    -------
    float
        Optimal spend level, or np.nan if target not achievable
    """
    def roi_diff(spend):
        if spend <= 0:
            return np.inf
        response = response_function(spend)
        roi = response / spend
        return abs(roi - target_roi)
    
    result = minimize(
        roi_diff,
        x0=[(spend_range[0] + spend_range[1]) / 2],
        bounds=[spend_range],
        method='L-BFGS-B'
    )
    
    if result.success:
        optimal_spend = result.x[0]
        actual_roi = response_function(optimal_spend) / optimal_spend
        
        if abs(actual_roi - target_roi) < 0.01:  # Within 1% of target
            return optimal_spend
    
    return np.nan


def incremental_budget_impact(
    response_functions: Dict[str, Callable],
    current_allocation: Dict[str, float],
    budget_change: float,
    method: str = 'optimal'
) -> Dict:
    """
    Analyze impact of budget increase/decrease.
    
    Parameters
    ----------
    response_functions : dict
        Channel response functions
    current_allocation : dict
        Current budget allocation
    budget_change : float
        Change in total budget (positive or negative)
    method : str, default='optimal'
        How to allocate change: 'optimal', 'proportional', or 'marginal'
        
    Returns
    -------
    dict
        Impact analysis results
    """
    current_total = sum(current_allocation.values())
    new_total = current_total + budget_change
    
    if new_total < 0:
        raise ValueError("Budget change would result in negative total budget")
    
    # Calculate current response
    current_response = sum(
        response_functions[ch](current_allocation[ch])
        for ch in current_allocation.keys()
    )
    
    if method == 'optimal':
        # Re-optimize with new budget
        result = optimize_budget_allocation(
            response_functions,
            new_total,
            method='SLSQP'
        )
        new_allocation = result['optimal_allocation']
        new_response = result['total_response']
        
    elif method == 'proportional':
        # Scale all channels proportionally
        scale_factor = new_total / current_total
        new_allocation = {
            ch: current_allocation[ch] * scale_factor
            for ch in current_allocation.keys()
        }
        new_response = sum(
            response_functions[ch](new_allocation[ch])
            for ch in new_allocation.keys()
        )
        
    elif method == 'marginal':
        # Use marginal ROI approach
        result = marginal_roi_allocation(
            response_functions,
            new_total
        )
        new_allocation = result['optimal_allocation']
        new_response = result['total_response']
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate impacts
    response_change = new_response - current_response
    response_lift = (new_response / current_response - 1) * 100
    marginal_roi = response_change / budget_change if budget_change != 0 else 0
    
    allocation_changes = {
        ch: new_allocation[ch] - current_allocation.get(ch, 0)
        for ch in new_allocation.keys()
    }
    
    return {
        'current_allocation': current_allocation,
        'new_allocation': new_allocation,
        'allocation_changes': allocation_changes,
        'budget_change': budget_change,
        'current_response': current_response,
        'new_response': new_response,
        'response_change': response_change,
        'response_lift_pct': response_lift,
        'marginal_roi': marginal_roi
    }


if __name__ == "__main__":
    # Example usage
    from src.transformations import hill_transform, geometric_adstock
    
    # Create response functions based on transformed curves
    def create_channel_response(adstock_rate, alpha, k, base_efficiency):
        def response(spend):
            if spend <= 0:
                return 0
            # Apply adstock
            adstock_spend = spend * (1 / (1 - adstock_rate))  # Simplified
            # Apply saturation
            saturated = hill_transform(np.array([adstock_spend]), alpha, k)[0]
            return saturated * base_efficiency
        return response
    
    response_functions = {
        'TV': create_channel_response(0.7, 30000, 2.0, 50000),
        'Digital': create_channel_response(0.5, 20000, 2.5, 40000),
        'Print': create_channel_response(0.3, 15000, 1.8, 25000),
        'Radio': create_channel_response(0.6, 12000, 2.2, 30000)
    }
    
    # Optimize budget
    print("Optimizing budget allocation...")
    result = optimize_budget_allocation(
        response_functions,
        total_budget=100000
    )
    
    print("\nOptimal Allocation:")
    for ch, spend in result['optimal_allocation'].items():
        print(f"  {ch}: ${spend:,.0f}")
    
    print(f"\nTotal Expected Response: ${result['total_response']:,.0f}")
    print(f"\nROI by Channel:")
    for ch, roi in result['roi_by_channel'].items():
        print(f"  {ch}: {roi:.2f}x")
