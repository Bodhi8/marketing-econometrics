
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
