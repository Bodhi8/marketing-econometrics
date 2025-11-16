"""
Causal inference methods for marketing analytics

This module implements various causal inference techniques including:
- Difference-in-Differences (DiD)
- Regression Discontinuity (RD)
- Instrumental Variables (IV)
- Propensity Score Matching
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings


def difference_in_differences(df: pd.DataFrame,
                              outcome_var: str,
                              treatment_var: str,
                              unit_var: str,
                              time_var: str,
                              post_period: Optional[Union[int, str]] = None,
                              covariates: Optional[list] = None) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Perform Difference-in-Differences analysis.
    
    The DiD estimator compares the change in outcomes between treatment
    and control groups before and after treatment.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with units and time periods
    outcome_var : str
        Outcome variable name
    treatment_var : str
        Binary treatment group indicator (0/1)
    unit_var : str
        Unit identifier (e.g., market_id, customer_id)
    time_var : str
        Time period identifier
    post_period : int or str, optional
        Value of time_var indicating post-treatment period
        If None, assumes treatment_var indicates actual treatment
    covariates : list, optional
        Additional control variables to include
    
    Returns
    -------
    tuple
        (regression results, summary DataFrame)
    
    Examples
    --------
    >>> did_results, summary = difference_in_differences(
    ...     df, outcome_var='sales', treatment_var='treatment',
    ...     unit_var='market_id', time_var='week'
    ... )
    >>> print(f"Treatment Effect: ${summary.loc['DiD Estimate', 'value']:,.0f}")
    """
    df = df.copy()
    
    # Create post-treatment indicator if specified
    if post_period is not None:
        df['post'] = (df[time_var] >= post_period).astype(int)
    else:
        # Assume treatment_var indicates actual treatment in each period
        df['post'] = df.groupby(unit_var)[treatment_var].transform(lambda x: (x == 1).any()).astype(int)
        df['treatment_group'] = df.groupby(unit_var)[treatment_var].transform('max')
        treatment_var = 'treatment_group'
    
    # Create treatment indicator if not boolean
    df[treatment_var] = df[treatment_var].astype(int)
    
    # Build formula
    formula_parts = [
        f'{outcome_var} ~',
        f'{treatment_var}',
        '+ post',
        f'+ {treatment_var}:post',  # DiD interaction term
        f'+ C({unit_var})',  # Unit fixed effects
        f'+ C({time_var})'   # Time fixed effects
    ]
    
    # Add covariates
    if covariates:
        formula_parts.append('+ ' + ' + '.join(covariates))
    
    formula = ' '.join(formula_parts)
    
    # Fit model
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[unit_var]})
    
    # Extract DiD estimate
    interaction_term = f'{treatment_var}:post'
    did_estimate = model.params[interaction_term]
    did_se = model.bse[interaction_term]
    did_pvalue = model.pvalues[interaction_term]
    did_ci = model.conf_int().loc[interaction_term]
    
    # Create summary
    summary = pd.DataFrame({
        'metric': [
            'DiD Estimate',
            'Standard Error',
            'P-value',
            'CI Lower (95%)',
            'CI Upper (95%)',
            'Observations',
            'R-squared'
        ],
        'value': [
            did_estimate,
            did_se,
            did_pvalue,
            did_ci[0],
            did_ci[1],
            model.nobs,
            model.rsquared
        ]
    })
    
    return model, summary


def regression_discontinuity(df: pd.DataFrame,
                            outcome_var: str,
                            running_var: str,
                            cutoff: float,
                            bandwidth: Optional[float] = None,
                            polynomial_order: int = 1,
                            kernel: str = 'triangular') -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Perform Regression Discontinuity analysis.
    
    RD exploits discontinuous changes in treatment assignment based on
    a running variable crossing a cutoff threshold.
    
    Parameters
    ----------
    df : DataFrame
        Data with running variable and outcome
    outcome_var : str
        Outcome variable name
    running_var : str
        Running variable (assignment variable)
    cutoff : float
        Threshold value for treatment assignment
    bandwidth : float, optional
        Bandwidth around cutoff. If None, uses data-driven selection
    polynomial_order : int, default=1
        Order of polynomial in running variable (1=linear, 2=quadratic)
    kernel : str, default='triangular'
        Kernel weighting ('uniform', 'triangular', 'epanechnikov')
    
    Returns
    -------
    tuple
        (regression results, summary DataFrame)
    
    Examples
    --------
    >>> rd_results, summary = regression_discontinuity(
    ...     df, outcome_var='conversion', running_var='account_value',
    ...     cutoff=50000, bandwidth=10000
    ... )
    """
    df = df.copy()
    
    # Center running variable around cutoff
    df['running_centered'] = df[running_var] - cutoff
    
    # Create treatment indicator
    df['treatment'] = (df[running_var] >= cutoff).astype(int)
    
    # Select bandwidth if not provided
    if bandwidth is None:
        # Simple rule of thumb: use IQR/1.5
        iqr = df['running_centered'].abs().quantile(0.75) - df['running_centered'].abs().quantile(0.25)
        bandwidth = iqr / 1.5
        print(f"Data-driven bandwidth: {bandwidth:.2f}")
    
    # Filter to bandwidth
    df_rd = df[df['running_centered'].abs() <= bandwidth].copy()
    
    if len(df_rd) < 30:
        warnings.warn(f"Only {len(df_rd)} observations within bandwidth. Consider increasing bandwidth.")
    
    # Apply kernel weights
    if kernel == 'triangular':
        df_rd['weight'] = 1 - df_rd['running_centered'].abs() / bandwidth
    elif kernel == 'epanechnikov':
        df_rd['weight'] = 0.75 * (1 - (df_rd['running_centered'] / bandwidth)**2)
    elif kernel == 'uniform':
        df_rd['weight'] = 1.0
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Build formula with polynomial
    formula_parts = [f'{outcome_var} ~ treatment']
    
    for order in range(1, polynomial_order + 1):
        # Separate slopes before and after cutoff
        df_rd[f'running_centered_{order}'] = df_rd['running_centered'] ** order
        formula_parts.append(f'running_centered_{order}')
        formula_parts.append(f'treatment:running_centered_{order}')
    
    formula = ' + '.join(formula_parts)
    
    # Fit weighted regression
    model = smf.wls(formula, data=df_rd, weights=df_rd['weight']).fit(
        cov_type='HC3'  # Robust standard errors
    )
    
    # Extract treatment effect
    rd_estimate = model.params['treatment']
    rd_se = model.bse['treatment']
    rd_pvalue = model.pvalues['treatment']
    rd_ci = model.conf_int().loc['treatment']
    
    # Create summary
    summary = pd.DataFrame({
        'metric': [
            'RD Estimate (Local Average Treatment Effect)',
            'Standard Error',
            'P-value',
            'CI Lower (95%)',
            'CI Upper (95%)',
            'Bandwidth',
            'Observations (within bandwidth)',
            'Polynomial Order',
            'R-squared'
        ],
        'value': [
            rd_estimate,
            rd_se,
            rd_pvalue,
            rd_ci[0],
            rd_ci[1],
            bandwidth,
            len(df_rd),
            polynomial_order,
            model.rsquared
        ]
    })
    
    return model, summary


def instrumental_variables(df: pd.DataFrame,
                          outcome_var: str,
                          treatment_var: str,
                          instrument_var: Union[str, list],
                          covariates: Optional[list] = None) -> Tuple:
    """
    Perform Instrumental Variables (2SLS) estimation.
    
    IV addresses endogeneity by using instruments that affect treatment
    but not the outcome except through treatment.
    
    Parameters
    ----------
    df : DataFrame
        Data containing outcome, treatment, and instruments
    outcome_var : str
        Outcome variable name
    treatment_var : str
        Endogenous treatment variable
    instrument_var : str or list
        Instrumental variable(s)
    covariates : list, optional
        Exogenous control variables
    
    Returns
    -------
    tuple
        (first stage results, second stage results, summary DataFrame)
    
    Examples
    --------
    >>> fs, ss, summary = instrumental_variables(
    ...     df, outcome_var='sales', treatment_var='price',
    ...     instrument_var='cost_shock', covariates=['seasonality']
    ... )
    """
    from statsmodels.sandbox.regression.gmm import IV2SLS
    
    df = df.copy()
    
    # Prepare variables
    if isinstance(instrument_var, str):
        instrument_var = [instrument_var]
    
    # Build variable lists
    exog_vars = covariates if covariates else []
    
    # Prepare matrices
    y = df[outcome_var].values
    X = df[[treatment_var] + exog_vars].values if exog_vars else df[[treatment_var]].values
    Z = df[instrument_var + exog_vars].values if exog_vars else df[instrument_var].values
    
    # Add constant
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)
    
    # First stage: Regress treatment on instruments
    first_stage = sm.OLS(df[treatment_var], Z).fit()
    
    # Check instrument strength (F-statistic)
    f_stat = first_stage.fvalue
    
    if f_stat < 10:
        warnings.warn(f"Weak instruments detected (F={f_stat:.2f}). F > 10 recommended.")
    
    # Second stage: 2SLS estimation
    iv_model = IV2SLS(y, X, Z).fit()
    
    # Extract estimates
    treatment_effect = iv_model.params[1]  # First non-constant parameter
    treatment_se = iv_model.bse[1]
    treatment_pvalue = iv_model.pvalues[1]
    treatment_ci = iv_model.conf_int()[1]
    
    # Create summary
    summary = pd.DataFrame({
        'metric': [
            'IV Estimate',
            'Standard Error',
            'P-value',
            'CI Lower (95%)',
            'CI Upper (95%)',
            'First Stage F-statistic',
            'Observations',
            'Instrument Strength'
        ],
        'value': [
            treatment_effect,
            treatment_se,
            treatment_pvalue,
            treatment_ci[0],
            treatment_ci[1],
            f_stat,
            len(df),
            'Strong' if f_stat >= 10 else 'Weak'
        ]
    })
    
    return first_stage, iv_model, summary


def propensity_score_matching(df: pd.DataFrame,
                              outcome_var: str,
                              treatment_var: str,
                              covariates: list,
                              matching_method: str = 'nearest',
                              caliper: Optional[float] = None) -> pd.DataFrame:
    """
    Perform Propensity Score Matching.
    
    PSM creates comparable treatment and control groups by matching on
    the probability of receiving treatment.
    
    Parameters
    ----------
    df : DataFrame
        Data with treatment and covariates
    outcome_var : str
        Outcome variable name
    treatment_var : str
        Binary treatment indicator (0/1)
    covariates : list
        Variables to use in propensity score model
    matching_method : str, default='nearest'
        'nearest' or 'kernel'
    caliper : float, optional
        Maximum propensity score distance for matching
    
    Returns
    -------
    DataFrame
        Results including ATT, standard error, and matched sample
    
    Examples
    --------
    >>> psm_results = propensity_score_matching(
    ...     df, outcome_var='conversion', treatment_var='email_campaign',
    ...     covariates=['age', 'income', 'prior_purchases']
    ... )
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    
    df = df.copy()
    
    # Estimate propensity scores
    X = df[covariates]
    treatment = df[treatment_var]
    
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, treatment)
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    # Check common support
    ps_treated = df[df[treatment_var] == 1]['propensity_score']
    ps_control = df[df[treatment_var] == 0]['propensity_score']
    
    common_support_min = max(ps_treated.min(), ps_control.min())
    common_support_max = min(ps_treated.max(), ps_control.max())
    
    df_matched = df[
        (df['propensity_score'] >= common_support_min) &
        (df['propensity_score'] <= common_support_max)
    ].copy()
    
    print(f"Common support: [{common_support_min:.4f}, {common_support_max:.4f}]")
    print(f"Observations in common support: {len(df_matched)} / {len(df)}")
    
    if matching_method == 'nearest':
        # Nearest neighbor matching
        treated = df_matched[df_matched[treatment_var] == 1]
        control = df_matched[df_matched[treatment_var] == 0]
        
        # Find nearest neighbor for each treated unit
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(control[['propensity_score']])
        
        distances, indices = nn.kneighbors(treated[['propensity_score']])
        
        # Apply caliper if specified
        if caliper:
            valid_matches = distances.flatten() <= caliper
            treated = treated[valid_matches]
            indices = indices[valid_matches]
            print(f"Matches within caliper: {valid_matches.sum()} / {len(valid_matches)}")
        
        # Get matched control outcomes
        matched_control = control.iloc[indices.flatten()]
        
        # Calculate ATT
        att = (treated[outcome_var].values - matched_control[outcome_var].values).mean()
        att_se = (treated[outcome_var].values - matched_control[outcome_var].values).std() / np.sqrt(len(treated))
        
    else:
        raise NotImplementedError(f"Matching method '{matching_method}' not yet implemented")
    
    # Create results
    results = pd.DataFrame({
        'metric': [
            'ATT (Average Treatment Effect on Treated)',
            'Standard Error',
            'T-statistic',
            'P-value',
            'Matched Treated Units',
            'Matched Control Units',
            'Common Support Range'
        ],
        'value': [
            att,
            att_se,
            att / att_se,
            2 * (1 - stats.norm.cdf(abs(att / att_se))),
            len(treated),
            len(matched_control),
            f"[{common_support_min:.4f}, {common_support_max:.4f}]"
        ]
    })
    
    return results


def synthetic_control(df: pd.DataFrame,
                     outcome_var: str,
                     unit_var: str,
                     time_var: str,
                     treated_unit: Union[int, str],
                     treatment_time: Union[int, str],
                     donor_pool: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic Control Method for causal inference.
    
    Creates a synthetic version of the treated unit using a weighted
    combination of control units.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    outcome_var : str
        Outcome variable name
    unit_var : str
        Unit identifier
    time_var : str
        Time period identifier
    treated_unit : int or str
        ID of treated unit
    treatment_time : int or str
        Time when treatment begins
    donor_pool : list, optional
        List of unit IDs to use as donors. If None, uses all untreated units
    
    Returns
    -------
    tuple
        (results DataFrame, weights DataFrame)
    """
    from scipy.optimize import minimize
    
    df = df.copy()
    
    # Pre-treatment data
    df_pre = df[df[time_var] < treatment_time].copy()
    
    # Treated unit pre-treatment outcomes
    treated_pre = df_pre[df_pre[unit_var] == treated_unit][outcome_var].values
    
    # Donor pool
    if donor_pool is None:
        donor_pool = [u for u in df[unit_var].unique() if u != treated_unit]
    
    # Create donor matrix (time x units)
    donor_outcomes = np.column_stack([
        df_pre[df_pre[unit_var] == u][outcome_var].values
        for u in donor_pool
    ])
    
    # Optimize weights to match pre-treatment period
    def objective(weights):
        synthetic = donor_outcomes @ weights
        return np.sum((treated_pre - synthetic) ** 2)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * len(donor_pool)
    initial_weights = np.ones(len(donor_pool)) / len(donor_pool)
    
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # Calculate synthetic control for all periods
    synthetic_outcomes = []
    treated_outcomes = []
    time_periods = []
    
    for t in df[time_var].unique():
        df_t = df[df[time_var] == t]
        
        treated_outcome = df_t[df_t[unit_var] == treated_unit][outcome_var].values[0]
        
        donor_outcomes_t = np.array([
            df_t[df_t[unit_var] == u][outcome_var].values[0]
            for u in donor_pool
        ])
        
        synthetic_outcome = donor_outcomes_t @ optimal_weights
        
        time_periods.append(t)
        treated_outcomes.append(treated_outcome)
        synthetic_outcomes.append(synthetic_outcome)
    
    # Create results
    results = pd.DataFrame({
        time_var: time_periods,
        'treated': treated_outcomes,
        'synthetic': synthetic_outcomes,
        'gap': np.array(treated_outcomes) - np.array(synthetic_outcomes),
        'post_treatment': [t >= treatment_time for t in time_periods]
    })
    
    # Calculate treatment effect (average post-treatment gap)
    att = results[results['post_treatment']]['gap'].mean()
    
    # Weights
    weights_df = pd.DataFrame({
        'unit': donor_pool,
        'weight': optimal_weights
    }).sort_values('weight', ascending=False)
    
    print(f"Average Treatment Effect (post-treatment): {att:.2f}")
    print(f"Pre-treatment RMSPE: {np.sqrt(objective(optimal_weights)):.2f}")
    
    return results, weights_df
