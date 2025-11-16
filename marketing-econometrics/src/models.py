"""
Marketing Mix Models (MMM) implementations.

This module provides various approaches to MMM including:
- Bayesian MMM using PyMC
- Regularized regression models
- Model validation utilities
- Attribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

# Scikit-learn imports
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# PyMC for Bayesian modeling
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not installed. Bayesian models will not be available.")


class BaseMMM:
    """Base class for Marketing Mix Models."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def _validate_inputs(self, X, y):
        """Validate input data."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")
        return True
    
    def get_channel_contributions(self, X: np.ndarray) -> pd.DataFrame:
        """
        Calculate contribution of each channel to total predicted sales.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (channels × time)
            
        Returns
        -------
        pd.DataFrame
            Contributions by channel
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating contributions")
        
        raise NotImplementedError("Subclasses must implement this method")


class RidgeMMM(BaseMMM):
    """
    Ridge Regression Marketing Mix Model.
    
    Uses L2 regularization to handle multicollinearity between channels.
    Good for stable parameter estimates when channels are correlated.
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        """
        Initialize Ridge MMM.
        
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength. Higher values = more regularization
        fit_intercept : bool, default=True
            Whether to fit intercept (baseline sales)
        """
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None):
        """
        Fit the Ridge MMM.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target variable (sales)
        feature_names : list, optional
            Names of features/channels
        """
        self._validate_inputs(X, y)
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"channel_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sales."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coefs = pd.DataFrame({
            'channel': self.feature_names,
            'coefficient': self.model.coef_
        })
        
        if self.fit_intercept:
            coefs = pd.concat([
                pd.DataFrame({'channel': ['intercept'], 
                            'coefficient': [self.model.intercept_]}),
                coefs
            ])
        
        return coefs
    
    def get_channel_contributions(self, X: np.ndarray) -> pd.DataFrame:
        """
        Calculate contribution of each channel.
        
        Contribution = coefficient × input value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Calculate contributions
        contributions = X * self.model.coef_
        
        contrib_df = pd.DataFrame(
            contributions,
            columns=self.feature_names
        )
        
        # Add baseline if intercept was fitted
        if self.fit_intercept:
            contrib_df['baseline'] = self.model.intercept_
        
        return contrib_df
    
    def calculate_roi(self, X: np.ndarray, spend: np.ndarray) -> pd.DataFrame:
        """
        Calculate ROI for each channel.
        
        Parameters
        ----------
        X : np.ndarray
            Transformed features used in model
        spend : np.ndarray
            Actual spend amounts (same shape as X)
            
        Returns
        -------
        pd.DataFrame
            ROI metrics by channel
        """
        contributions = self.get_channel_contributions(X)
        
        roi_data = []
        for i, channel in enumerate(self.feature_names):
            total_contribution = contributions[channel].sum()
            total_spend = spend[:, i].sum()
            
            if total_spend > 0:
                roi = total_contribution / total_spend
                efficiency = total_contribution / contributions[channel].count()
            else:
                roi = 0
                efficiency = 0
            
            roi_data.append({
                'channel': channel,
                'total_spend': total_spend,
                'total_contribution': total_contribution,
                'roi': roi,
                'avg_efficiency': efficiency
            })
        
        return pd.DataFrame(roi_data)


class BayesianMMM(BaseMMM):
    """
    Bayesian Marketing Mix Model using PyMC.
    
    Provides full posterior distributions for parameters, enabling:
    - Uncertainty quantification
    - Informative priors from business knowledge
    - Hierarchical structures
    """
    
    def __init__(self, n_samples: int = 2000, n_tune: int = 1000):
        """
        Initialize Bayesian MMM.
        
        Parameters
        ----------
        n_samples : int, default=2000
            Number of posterior samples to draw
        n_tune : int, default=1000
            Number of tuning steps for sampler
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required for BayesianMMM")
        
        super().__init__()
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.trace = None
        self.posterior_predictive = None
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            priors: Optional[Dict] = None):
        """
        Fit Bayesian MMM.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : list, optional
            Channel names
        priors : dict, optional
            Prior distributions for parameters
            Example: {'beta_mu': 0, 'beta_sigma': 10}
        """
        self._validate_inputs(X, y)
        
        if feature_names is None:
            feature_names = [f"channel_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        
        # Default priors
        if priors is None:
            priors = {
                'intercept_mu': y.mean(),
                'intercept_sigma': y.std() * 2,
                'beta_mu': 0,
                'beta_sigma': y.std(),
                'sigma_alpha': 2,
                'sigma_beta': 0.5
            }
        
        n_features = X.shape[1]
        
        with pm.Model() as self.model:
            # Priors for coefficients
            intercept = pm.Normal('intercept', 
                                 mu=priors['intercept_mu'],
                                 sigma=priors['intercept_sigma'])
            
            # Positive coefficients for marketing channels (half-normal)
            beta = pm.HalfNormal('beta', 
                                sigma=priors['beta_sigma'],
                                shape=n_features)
            
            # Error term
            sigma = pm.HalfCauchy('sigma',
                                 beta=priors['sigma_beta'])
            
            # Expected value
            mu = intercept + pm.math.dot(X, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample from posterior
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                return_inferencedata=True,
                progressbar=False
            )
            
            # Posterior predictive samples
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                progressbar=False
            )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Predict with uncertainty.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict
        return_std : bool, default=False
            If True, return (mean, std) of predictions
            
        Returns
        -------
        predictions : np.ndarray or tuple
            Mean predictions, or (mean, std) if return_std=True
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get posterior means
        intercept_mean = self.trace.posterior['intercept'].mean().values
        beta_mean = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        
        # Predict
        y_pred_mean = intercept_mean + X @ beta_mean
        
        if return_std:
            # Calculate prediction uncertainty
            intercept_samples = self.trace.posterior['intercept'].values.flatten()
            beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_names))
            
            # Monte Carlo predictions
            predictions = intercept_samples[:, None] + X @ beta_samples.T
            y_pred_std = predictions.std(axis=0)
            
            return y_pred_mean, y_pred_std
        
        return y_pred_mean
    
    def get_coefficients(self, credible_interval: float = 0.94) -> pd.DataFrame:
        """
        Get coefficient estimates with credible intervals.
        
        Parameters
        ----------
        credible_interval : float, default=0.94
            Width of credible interval (e.g., 0.94 for 94% CI)
            
        Returns
        -------
        pd.DataFrame
            Coefficient estimates with uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        summary = az.summary(self.trace, hdi_prob=credible_interval)
        
        coef_data = []
        
        # Intercept
        coef_data.append({
            'channel': 'intercept',
            'mean': summary.loc['intercept', 'mean'],
            'sd': summary.loc['intercept', 'sd'],
            f'hdi_{(1-credible_interval)/2:.1%}': summary.loc['intercept', f'hdi_{(1-credible_interval)/2:.1%}'],
            f'hdi_{1-(1-credible_interval)/2:.1%}': summary.loc['intercept', f'hdi_{1-(1-credible_interval)/2:.1%}']
        })
        
        # Channel coefficients
        for i, channel in enumerate(self.feature_names):
            coef_data.append({
                'channel': channel,
                'mean': summary.loc[f'beta[{i}]', 'mean'],
                'sd': summary.loc[f'beta[{i}]', 'sd'],
                f'hdi_{(1-credible_interval)/2:.1%}': summary.loc[f'beta[{i}]', f'hdi_{(1-credible_interval)/2:.1%}'],
                f'hdi_{1-(1-credible_interval)/2:.1%}': summary.loc[f'beta[{i}]', f'hdi_{1-(1-credible_interval)/2:.1%}']
            })
        
        return pd.DataFrame(coef_data)
    
    def get_channel_contributions(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate channel contributions using posterior mean coefficients."""
        beta_mean = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        intercept_mean = self.trace.posterior['intercept'].mean().values
        
        contributions = X * beta_mean
        contrib_df = pd.DataFrame(contributions, columns=self.feature_names)
        contrib_df['baseline'] = intercept_mean
        
        return contrib_df


def cross_validate_mmm(model_class, X: np.ndarray, y: np.ndarray,
                       n_splits: int = 5, **model_kwargs) -> Dict:
    """
    Time series cross-validation for MMM.
    
    Parameters
    ----------
    model_class : class
        MMM class to use (e.g., RidgeMMM)
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    n_splits : int, default=5
        Number of CV splits
    **model_kwargs
        Arguments to pass to model constructor
        
    Returns
    -------
    dict
        Cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = {
        'r2': [],
        'mae': [],
        'rmse': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores['r2'].append(r2_score(y_test, y_pred))
        scores['mae'].append(mean_absolute_error(y_test, y_pred))
        scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return {
        'r2_mean': np.mean(scores['r2']),
        'r2_std': np.std(scores['r2']),
        'mae_mean': np.mean(scores['mae']),
        'mae_std': np.std(scores['mae']),
        'rmse_mean': np.mean(scores['rmse']),
        'rmse_std': np.std(scores['rmse'])
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100
    n_channels = 3
    
    X = np.random.randn(n_samples, n_channels) * 10 + 50
    true_coefs = np.array([1.5, 2.0, 1.0])
    y = 1000 + X @ true_coefs + np.random.randn(n_samples) * 50
    
    # Fit Ridge MMM
    print("Fitting Ridge MMM...")
    ridge_model = RidgeMMM(alpha=1.0)
    ridge_model.fit(X, y, feature_names=['TV', 'Digital', 'Print'])
    
    print("\nCoefficients:")
    print(ridge_model.get_coefficients())
    
    print("\nR² score:", r2_score(y, ridge_model.predict(X)))
