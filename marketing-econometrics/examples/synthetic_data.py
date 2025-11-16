"""
Generate synthetic marketing data for examples and testing

This script creates realistic marketing data with known relationships
between marketing activities and outcomes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_marketing_data(n_weeks=104, seed=42):
    """
    Generate synthetic marketing data.
    
    Parameters
    ----------
    n_weeks : int, default=104
        Number of weeks of data (default is 2 years)
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    DataFrame
        Synthetic marketing data
    """
    np.random.seed(seed)
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
    
    # Marketing channels with realistic spend patterns
    # TV: High spend, seasonal peaks
    tv_base = 80000
    tv_seasonal = 30000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    tv_trend = np.linspace(0, 20000, n_weeks)
    tv_spend = tv_base + tv_seasonal + tv_trend + np.random.normal(0, 10000, n_weeks)
    tv_spend = np.maximum(tv_spend, 20000)  # Floor
    
    # Digital: Growing spend, less seasonal
    digital_base = 50000
    digital_trend = np.linspace(0, 40000, n_weeks)
    digital_spend = digital_base + digital_trend + np.random.normal(0, 8000, n_weeks)
    digital_spend = np.maximum(digital_spend, 15000)
    
    # Radio: Moderate spend, stable
    radio_spend = np.random.uniform(15000, 35000, n_weeks)
    
    # Print: Declining spend
    print_base = 30000
    print_trend = -np.linspace(0, 15000, n_weeks)
    print_spend = print_base + print_trend + np.random.normal(0, 5000, n_weeks)
    print_spend = np.maximum(print_spend, 5000)
    
    # OOH (Out-of-Home): Sporadic campaigns
    ooh_spend = np.zeros(n_weeks)
    campaign_weeks = [10, 11, 12, 30, 31, 32, 60, 61, 62, 90, 91, 92]
    ooh_spend[campaign_weeks] = np.random.uniform(40000, 60000, len(campaign_weeks))
    
    # Control variables
    # Price: Mean-reverting with some seasonality
    price_base = 99.99
    price_seasonal = 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + np.pi/2)
    price = price_base + price_seasonal + np.random.normal(0, 2, n_weeks)
    
    # Competitor price
    competitor_price = price + np.random.normal(5, 3, n_weeks)
    
    # Distribution (number of stores/outlets)
    distribution = 500 + np.linspace(0, 100, n_weeks) + np.random.normal(0, 10, n_weeks)
    distribution = distribution.astype(int)
    
    # Promotions (binary indicator)
    promotion = np.random.choice([0, 1], n_weeks, p=[0.7, 0.3])
    
    # Macroeconomic indicator (consumer confidence)
    consumer_confidence = 100 + 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + \
                         np.cumsum(np.random.normal(0, 1, n_weeks))
    
    # Generate sales with known relationships
    # Apply adstock and saturation implicitly in the relationship
    import sys
    sys.path.append('..')
    from src.transformations import geometric_adstock, hill_saturation
    
    # Base sales
    baseline = 500000
    
    # Seasonality
    seasonality = 100000 * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52))
    
    # TV effect (strong carryover, high saturation point)
    tv_adstocked = geometric_adstock(tv_spend, theta=0.5)
    tv_saturated = hill_saturation(tv_adstocked, alpha=100000, k=2)
    tv_contribution = 150000 * tv_saturated
    
    # Digital effect (moderate carryover, moderate saturation)
    digital_adstocked = geometric_adstock(digital_spend, theta=0.3)
    digital_saturated = hill_saturation(digital_adstocked, alpha=60000, k=1.5)
    digital_contribution = 120000 * digital_saturated
    
    # Radio effect (low carryover)
    radio_adstocked = geometric_adstock(radio_spend, theta=0.2)
    radio_saturated = hill_saturation(radio_adstocked, alpha=25000, k=1)
    radio_contribution = 80000 * radio_saturated
    
    # Print effect (low contribution)
    print_adstocked = geometric_adstock(print_spend, theta=0.1)
    print_saturated = hill_saturation(print_adstocked, alpha=20000, k=1)
    print_contribution = 40000 * print_saturated
    
    # OOH effect (immediate, high impact)
    ooh_saturated = hill_saturation(ooh_spend, alpha=50000, k=1.5)
    ooh_contribution = 60000 * ooh_saturated
    
    # Price elasticity
    price_elasticity = -8000 * (price - price_base) / price_base
    
    # Competitor price effect
    comp_price_effect = 5000 * (competitor_price - price) / price
    
    # Distribution effect
    distribution_effect = 300 * (distribution - 500)
    
    # Promotion effect
    promotion_effect = promotion * 50000
    
    # Consumer confidence effect
    confidence_effect = 2000 * (consumer_confidence - 100)
    
    # Combine all effects
    sales = (
        baseline +
        seasonality +
        tv_contribution +
        digital_contribution +
        radio_contribution +
        print_contribution +
        ooh_contribution +
        price_elasticity +
        comp_price_effect +
        distribution_effect +
        promotion_effect +
        confidence_effect +
        np.random.normal(0, 20000, n_weeks)  # Noise
    )
    
    # Ensure non-negative sales
    sales = np.maximum(sales, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'tv_spend': tv_spend,
        'digital_spend': digital_spend,
        'radio_spend': radio_spend,
        'print_spend': print_spend,
        'ooh_spend': ooh_spend,
        'price': price,
        'competitor_price': competitor_price,
        'distribution': distribution,
        'promotion': promotion,
        'consumer_confidence': consumer_confidence,
    })
    
    # Add time features
    df['week'] = range(1, n_weeks + 1)
    df['week_of_year'] = [d.isocalendar()[1] for d in dates]
    df['month'] = [d.month for d in dates]
    df['quarter'] = [((d.month - 1) // 3) + 1 for d in dates]
    
    # Add sine/cosine encoding for seasonality
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Add year indicator
    df['year'] = [d.year for d in dates]
    
    return df


def generate_geo_experiment_data(n_markets=30, n_weeks=24, treatment_week=13, seed=42):
    """
    Generate synthetic geo-experiment data for DiD analysis.
    
    Parameters
    ----------
    n_markets : int, default=30
        Number of geographic markets
    n_weeks : int, default=24
        Number of weeks
    treatment_week : int, default=13
        Week when treatment begins
    seed : int, default=42
        Random seed
    
    Returns
    -------
    DataFrame
        Geo-experiment data
    """
    np.random.seed(seed)
    
    data = []
    treatment_effect = 25000  # True treatment effect
    
    for market_id in range(n_markets):
        # Random assignment to treatment
        is_treatment = market_id < n_markets // 2
        
        # Market-specific baseline
        market_baseline = np.random.uniform(80000, 120000)
        
        # Market-specific trend
        market_trend = np.random.uniform(-500, 500)
        
        for week in range(1, n_weeks + 1):
            # Is this market treated in this week?
            treated = is_treatment and (week >= treatment_week)
            
            sales = (
                market_baseline +
                market_trend * week +
                (treatment_effect if treated else 0) +
                np.random.normal(0, 8000)
            )
            
            data.append({
                'market_id': market_id,
                'week': week,
                'sales': sales,
                'treatment': int(is_treatment),
                'treated': int(treated),
                'market_size': np.random.choice(['small', 'medium', 'large'])
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate and save marketing data
    print("Generating marketing data...")
    df_marketing = generate_marketing_data(n_weeks=104)
    df_marketing.to_csv('../data/sample_data.csv', index=False)
    print(f"Saved sample_data.csv ({len(df_marketing)} rows)")
    
    # Generate and save geo-experiment data
    print("\nGenerating geo-experiment data...")
    df_geo = generate_geo_experiment_data(n_markets=30, n_weeks=24)
    df_geo.to_csv('../data/geo_experiment_data.csv', index=False)
    print(f"Saved geo_experiment_data.csv ({len(df_geo)} rows)")
    
    # Display summary statistics
    print("\nMarketing Data Summary:")
    print(df_marketing[['sales', 'tv_spend', 'digital_spend', 'radio_spend', 'print_spend']].describe())
    
    print("\nGeo-Experiment Data Summary:")
    print(df_geo.groupby(['treatment', 'treated'])['sales'].describe())
