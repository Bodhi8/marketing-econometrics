"""
Generate synthetic Marketing Mix Modeling data with realistic patterns.

This script creates a dataset with:
- Multiple marketing channels (TV, Digital, Print, Radio)
- Adstock effects (carryover)
- Saturation effects (diminishing returns)
- Seasonality
- Control variables
- Realistic noise
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_weeks = 104  # 2 years of data
start_date = datetime(2023, 1, 1)

def geometric_adstock(x, decay_rate):
    """Apply geometric adstock transformation."""
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay_rate * adstocked[t-1]
    return adstocked

def hill_saturation(x, alpha, k):
    """Apply Hill saturation transformation."""
    return x**k / (x**k + alpha**k)

# Generate date range
dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

# Generate base marketing spend with trends and noise
tv_spend = 30000 + 10000 * np.sin(np.linspace(0, 4*np.pi, n_weeks)) + np.random.normal(0, 3000, n_weeks)
tv_spend = np.maximum(tv_spend, 5000)  # Ensure positive

digital_spend = 20000 + 8000 * np.sin(np.linspace(0.5, 4.5*np.pi, n_weeks)) + np.random.normal(0, 2500, n_weeks)
digital_spend = np.maximum(digital_spend, 3000)

print_spend = 15000 + 5000 * np.cos(np.linspace(0, 4*np.pi, n_weeks)) + np.random.normal(0, 2000, n_weeks)
print_spend = np.maximum(print_spend, 2000)

radio_spend = 12000 + 4000 * np.sin(np.linspace(0.25, 4.25*np.pi, n_weeks)) + np.random.normal(0, 1500, n_weeks)
radio_spend = np.maximum(radio_spend, 2000)

# Apply transformations to simulate realistic marketing effects

# TV: Strong adstock (70% retention), moderate saturation
tv_adstock = geometric_adstock(tv_spend, decay_rate=0.7)
tv_transformed = hill_saturation(tv_adstock, alpha=30000, k=2)

# Digital: Moderate adstock (50% retention), strong saturation
digital_adstock = geometric_adstock(digital_spend, decay_rate=0.5)
digital_transformed = hill_saturation(digital_adstock, alpha=20000, k=2.5)

# Print: Low adstock (30% retention), moderate saturation
print_adstock = geometric_adstock(print_spend, decay_rate=0.3)
print_transformed = hill_saturation(print_adstock, alpha=15000, k=1.8)

# Radio: Moderate adstock (60% retention), strong saturation
radio_adstock = geometric_adstock(radio_spend, decay_rate=0.6)
radio_transformed = hill_saturation(radio_adstock, alpha=12000, k=2.2)

# Generate control variables

# Price (inverse relationship with sales)
base_price = 50
price = base_price + 5 * np.sin(np.linspace(0, 2*np.pi, n_weeks)) + np.random.normal(0, 1, n_weeks)

# Promotions (binary indicator, happens ~25% of weeks)
promotion = np.random.binomial(1, 0.25, n_weeks)

# Seasonality (peaks in Q4, dips in summer)
week_of_year = np.array([d.isocalendar()[1] for d in dates])
seasonality = 1 + 0.3 * np.sin(2 * np.pi * (week_of_year - 10) / 52)

# Competitor activity (negative effect on sales)
competitor_activity = 5000 + 2000 * np.sin(np.linspace(0.3, 4.3*np.pi, n_weeks)) + np.random.normal(0, 500, n_weeks)
competitor_activity = np.maximum(competitor_activity, 1000)

# Generate sales based on all factors
baseline_sales = 50000

# Channel contributions (scaled by realistic ROI)
tv_contribution = 0.8 * tv_transformed
digital_contribution = 1.2 * digital_transformed
print_contribution = 0.4 * print_transformed
radio_contribution = 0.6 * radio_transformed

# Control effects
price_effect = -1000 * (price - base_price)
promotion_effect = 8000 * promotion
seasonality_effect = 20000 * (seasonality - 1)
competitor_effect = -0.3 * (competitor_activity - 5000)

# Combine all effects
sales = (baseline_sales + 
         tv_contribution + 
         digital_contribution + 
         print_contribution + 
         radio_contribution +
         price_effect +
         promotion_effect +
         seasonality_effect +
         competitor_effect +
         np.random.normal(0, 3000, n_weeks))  # Noise

# Ensure positive sales
sales = np.maximum(sales, 10000)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'tv_spend': tv_spend,
    'digital_spend': digital_spend,
    'print_spend': print_spend,
    'radio_spend': radio_spend,
    'price': price,
    'promotion': promotion,
    'competitor_activity': competitor_activity,
    'sales': sales
})

# Add derived columns
df['week'] = range(1, n_weeks + 1)
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year

# Calculate some useful metrics
df['total_spend'] = df['tv_spend'] + df['digital_spend'] + df['print_spend'] + df['radio_spend']
df['sales_to_spend_ratio'] = df['sales'] / df['total_spend']

# Save to CSV
output_path = 'data/synthetic_mmm_data.csv'
df.to_csv(output_path, index=False)

# Print summary statistics
print("Synthetic MMM Data Generated Successfully!")
print(f"Dataset saved to: {output_path}")
print(f"\nDataset shape: {df.shape}")
print(f"\nSummary Statistics:")
print(df.describe().round(2))

print(f"\nChannel Spend Ranges:")
print(f"TV: ${df['tv_spend'].min():.0f} - ${df['tv_spend'].max():.0f}")
print(f"Digital: ${df['digital_spend'].min():.0f} - ${df['digital_spend'].max():.0f}")
print(f"Print: ${df['print_spend'].min():.0f} - ${df['print_spend'].max():.0f}")
print(f"Radio: ${df['radio_spend'].min():.0f} - ${df['radio_spend'].max():.0f}")

print(f"\nSales Range: ${df['sales'].min():.0f} - ${df['sales'].max():.0f}")
print(f"Mean Sales: ${df['sales'].mean():.0f}")

# Save a data dictionary
data_dict = """
# Data Dictionary for synthetic_mmm_data.csv

## Date Variables
- **date**: Week ending date (Sunday)
- **week**: Week number (1-104)
- **month**: Month (1-12)
- **quarter**: Quarter (1-4)
- **year**: Year

## Marketing Spend Variables (USD)
- **tv_spend**: Weekly television advertising spend
- **digital_spend**: Weekly digital marketing spend (search, social, display)
- **print_spend**: Weekly print advertising spend (newspapers, magazines)
- **radio_spend**: Weekly radio advertising spend

## Control Variables
- **price**: Average product price for the week (USD)
- **promotion**: Binary indicator for promotional activity (0/1)
- **competitor_activity**: Estimated competitor marketing activity (USD)

## Outcome Variable
- **sales**: Weekly sales revenue (USD)

## Derived Variables
- **total_spend**: Sum of all marketing channel spend
- **sales_to_spend_ratio**: Efficiency metric (sales / total_spend)

## Data Generation Details

The data was generated to reflect realistic marketing dynamics:

1. **Adstock Effects**: Each channel has different carryover rates
   - TV: 70% retention (long-lasting effects)
   - Digital: 50% retention (moderate carryover)
   - Print: 30% retention (short-term effects)
   - Radio: 60% retention (good carryover)

2. **Saturation Effects**: Diminishing returns implemented via Hill transformation
   - Each channel has different saturation parameters
   - Higher spend shows progressively smaller marginal returns

3. **Seasonality**: Annual cycle with Q4 peak and summer dip

4. **Price Elasticity**: Negative relationship between price and sales

5. **Promotional Effects**: Significant sales lift during promotional weeks

6. **Competitive Effects**: Competitor activity negatively impacts sales

7. **Noise**: Realistic random variation added to all variables
"""

with open('data/DATA_DICTIONARY.md', 'w') as f:
    f.write(data_dict)

print("\nData dictionary saved to: data/DATA_DICTIONARY.md")
