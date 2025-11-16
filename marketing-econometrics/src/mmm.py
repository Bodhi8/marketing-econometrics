"""
Marketing Mix Modeling (MMM) implementation

Core classes for building and evaluating marketing mix models with
support for various transformation functions, regularization methods,
and diagnostic tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings

from .transformations import geometric_adstock, hill_saturation


class MarketingMixModel:
    """
    Marketing Mix Model with adstock and saturation transformations.
    
    This class implements a flexible MMM that can handle:
    - Multiple marketing channels
    - Custom adstock and saturation transformations
    - Regularization (Ridge, LASSO, ElasticNet)
    - Time-series cross-validation
    - Contribution decomposition
    
    Attributes
    ----------
    adstock_params : dict
        Channel-specific adstock parameters
    saturation_params : dict
        Channel-specific saturation parameters
    model_type : str
        Type of regression model ('ridge', 'lasso', 'elasticnet')
    alpha : float
        Regularization strength
    
    Examples
    --------
    >>> mmm = MarketingMixModel(
    ...     adstock_params={'tv': 0.5, 'digital': 0.3},
    ...     saturation_params={'tv': {'alpha': 100000, 'k': 2}},
    ...     model_type='ridge',
    ...     alpha=1.0
    ... )
    >>> mmm.fit(X_train, y_train, channel_names=['tv', 'digital', 'radio'])
    >>> predictions = mmm.predict(X_test)
    """
    
    def __init__(self, 
                 adstock_params: Optional[Dict[str, float]] = None,
                 saturation_params: Optional[Dict[str, Dict[str, float]]] = None,
                 model_type: str = 'ridge',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5):
        """
        Initialize the Marketing Mix Model.
        
        Parameters
        ----------
        adstock_params : dict, optional
            Adstock decay rates by channel, e.g. {'tv': 0.5, 'digital': 0.3}
        saturation_params : dict, optional
            Saturation parameters by channel, e.g. 
            {'tv': {'alpha': 100000, 'k': 2}}
        model_type : str, default='ridge'
            Type of regression: 'ridge', 'lasso', or 'elasticnet'
        alpha : float, default=1.0
            Regularization strength
        l1_ratio : float, default=0.5
            ElasticNet mixing parameter (only used if model_type='elasticnet')
        """
        self.adstock_params = adstock_params or {}
        self.saturation_params = saturation_params or {}
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # Initialize model based on type
        if self.model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=10000)
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        else:
            raise ValueError(f"model_type must be 'ridge', 'lasso', or 'elasticnet', got '{model_type}'")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def transform_channel(self, x: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Apply adstock and saturation transformations to a channel.
        
        Parameters
        ----------
        x : np.ndarray
            Raw channel data
        channel_name : str
            Name of the channel
        
        Returns
        -------
        np.ndarray
            Transformed channel data
        """
        # Apply adstock
        if channel_name in self.adstock_params:
            x = geometric_adstock(x, self.adstock_params[channel_name])
        
        # Apply saturation
        if channel_name in self.saturation_params:
            params = self.saturation_params[channel_name]
            x = hill_saturation(x, params['alpha'], params['k'])
        
        return x
    
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            channel_names: List[str],
            verbose: bool = True) -> 'MarketingMixModel':
        """
        Fit the Marketing Mix Model.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix with marketing and control variables
        y : Series or array-like
            Target variable (sales, revenue, etc.)
        channel_names : list of str
            Names of marketing channels to transform
        verbose : bool, default=True
            Print fitting progress
        
        Returns
        -------
        self : MarketingMixModel
            Fitted model
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        
        # Store metadata
        self.channel_names = channel_names
        self.feature_names = X.columns.tolist()
        self.n_features = X.shape[1]
        
        # Transform marketing channels
        X_transformed = X.copy()
        for channel in channel_names:
            if channel in X.columns:
                if verbose:
                    print(f"Transforming {channel}...")
                X_transformed[channel] = self.transform_channel(
                    X[channel].values, 
                    channel
                )
            else:
                warnings.warn(f"Channel '{channel}' not found in features")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_transformed)
        
        # Fit model
        if verbose:
            print(f"Fitting {self.model_type} model...")
        self.model.fit(X_scaled, y)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_scaled)
        self.train_r2 = r2_score(y, train_pred)
        self.train_mape = mean_absolute_percentage_error(y, train_pred)
        self.train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        
        if verbose:
            print(f"\nTraining Performance:")
            print(f"  R²: {self.train_r2:.4f}")
            print(f"  MAPE: {self.train_mape:.2%}")
            print(f"  RMSE: {self.train_rmse:,.0f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Transform channels
        X_transformed = X.copy()
        for channel in self.channel_names:
            if channel in X.columns:
                X_transformed[channel] = self.transform_channel(
                    X[channel].values,
                    channel
                )
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_transformed)
        return self.model.predict(X_scaled)
    
    def get_coefficients(self, sort_by: str = 'abs') -> pd.DataFrame:
        """
        Get model coefficients.
        
        Parameters
        ----------
        sort_by : str, default='abs'
            How to sort coefficients: 'abs' (absolute value) or 'value'
        
        Returns
        -------
        DataFrame
            Coefficients with feature names
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        if sort_by == 'abs':
            coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        elif sort_by == 'value':
            coef_df = coef_df.sort_values('coefficient', ascending=False)
        
        return coef_df
    
    def contribution_analysis(self, 
                             X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Calculate each variable's contribution to the total outcome.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix
        y : Series or array-like
            Target variable
        
        Returns
        -------
        DataFrame
            Contribution decomposition by feature
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        if isinstance(y, pd.Series):
            y = y.values
        
        # Transform and scale
        X_transformed = X.copy()
        for channel in self.channel_names:
            if channel in X.columns:
                X_transformed[channel] = self.transform_channel(
                    X[channel].values,
                    channel
                )
        
        X_scaled = self.scaler.transform(X_transformed)
        
        # Calculate contributions
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            # Contribution = coefficient * sum of scaled feature values
            feature_contribution = (X_scaled[:, i] * self.model.coef_[i]).sum()
            contributions[feature] = feature_contribution
        
        # Add baseline (intercept contribution)
        baseline_contribution = self.model.intercept_ * len(X)
        contributions['baseline'] = baseline_contribution
        
        # Add actual total
        contributions['actual_total'] = y.sum()
        
        # Predicted total
        predicted_total = sum(contributions.values()) - contributions['actual_total']
        contributions['predicted_total'] = predicted_total
        
        # Create DataFrame
        contrib_df = pd.DataFrame({
            'feature': list(contributions.keys()),
            'contribution': list(contributions.values())
        })
        
        # Calculate percentages
        total = contrib_df[contrib_df['feature'] == 'predicted_total']['contribution'].values[0]
        contrib_df['percentage'] = (contrib_df['contribution'] / total * 100)
        
        return contrib_df.sort_values('contribution', ascending=False)
    
    def cross_validate(self, 
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      channel_names: List[str],
                      n_splits: int = 5) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix
        y : Series or array-like
            Target variable
        channel_names : list of str
            Marketing channel names
        n_splits : int, default=5
            Number of CV splits
        
        Returns
        -------
        dict
            Cross-validation scores
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        
        # Transform channels
        X_transformed = X.copy()
        for channel in channel_names:
            if channel in X.columns:
                X_transformed[channel] = self.transform_channel(
                    X[channel].values,
                    channel
                )
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Perform CV
        r2_scores = []
        mape_scores = []
        
        for train_idx, val_idx in tscv.split(X_transformed):
            X_train, X_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Fit and predict
            model = self.model.__class__(**self.model.get_params())
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # Metrics
            r2_scores.append(r2_score(y_val, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))
        
        return {
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_mape': np.mean(mape_scores),
            'std_mape': np.std(mape_scores),
            'cv_scores': r2_scores
        }
    
    def response_curve(self, 
                      channel: str,
                      spend_range: Optional[Tuple[float, float]] = None,
                      n_points: int = 100) -> pd.DataFrame:
        """
        Generate response curve for a channel.
        
        Parameters
        ----------
        channel : str
            Channel name
        spend_range : tuple, optional
            (min, max) spend range. If None, uses (0, 2*max_observed)
        n_points : int, default=100
            Number of points to evaluate
        
        Returns
        -------
        DataFrame
            Response curve with spend and response columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if channel not in self.channel_names:
            raise ValueError(f"Channel '{channel}' not in model")
        
        # Get coefficient
        coef_idx = self.feature_names.index(channel)
        coef = self.model.coef_[coef_idx]
        
        # Generate spend range
        if spend_range is None:
            spend_range = (0, 200000)  # Default range
        
        spend_values = np.linspace(spend_range[0], spend_range[1], n_points)
        
        # Calculate responses
        responses = []
        for spend in spend_values:
            # Transform
            transformed = self.transform_channel(np.array([spend]), channel)[0]
            # Scale (approximately - using coefficient directly)
            response = coef * transformed
            responses.append(response)
        
        return pd.DataFrame({
            'spend': spend_values,
            'response': responses,
            'marginal_roi': np.gradient(responses, spend_values)
        })
    
    def summary(self) -> str:
        """
        Generate model summary.
        
        Returns
        -------
        str
            Formatted summary
        """
        if not self.is_fitted:
            return "Model not yet fitted"
        
        summary = f"""
Marketing Mix Model Summary
===========================

Model Type: {self.model_type.title()}
Regularization (α): {self.alpha}

Training Performance:
  R²: {self.train_r2:.4f}
  MAPE: {self.train_mape:.2%}
  RMSE: {self.train_rmse:,.0f}

Features: {self.n_features}
Marketing Channels: {len(self.channel_names)}
  {', '.join(self.channel_names)}

Transformations Applied:
  Adstock: {len(self.adstock_params)} channels
  Saturation: {len(self.saturation_params)} channels
        """
        return summary
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"MarketingMixModel(model_type='{self.model_type}', alpha={self.alpha}, status='{status}')"
