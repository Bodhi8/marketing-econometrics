Marketing Econometrics: A Practical Guide
This repository accompanies the Medium article "Marketing Econometrics: Unlocking Causal Intelligence in Marketing Strategy" and provides hands-on Python implementations of key marketing econometrics concepts.

Repository Structure
marketing-econometrics/
├── README.md
├── requirements.txt
├── data/
│   ├── synthetic_mmm_data.csv
│   └── data_generation.py
├── notebooks/
│   ├── 01_adstock_and_saturation.ipynb
│   ├── 02_marketing_mix_modeling.ipynb
│   ├── 03_causal_inference.ipynb
│   └── 04_optimization.ipynb
├── src/
│   ├── __init__.py
│   ├── transformations.py
│   ├── models.py
│   └── optimization.py
└── outputs/
    └── visualizations/
What's Included
1. Transformation Functions
Adstock transformations: Geometric, Weibull, and delayed adstock
Saturation curves: Hill transformation, logistic functions
Preprocessing utilities: Scaling, lag creation, feature engineering
2. Marketing Mix Models
Bayesian MMM: Using PyMC for full posterior inference
Regularized regression: Ridge, LASSO, and Elastic Net implementations
Model validation: Out-of-sample testing, cross-validation
Attribution analysis: Channel contribution decomposition
3. Causal Inference Methods
Difference-in-Differences: Standard and staggered adoption designs
Synthetic controls: Building counterfactual comparisons
Panel data methods: Fixed effects and first differences
4. Optimization
Budget allocation: Constrained optimization using scipy
Response curve optimization: Finding optimal spend levels
Scenario planning: What-if analysis tools
5. Visualizations
Response curves with confidence intervals
Attribution waterfall charts
ROI comparison plots
Time-series decomposition
Quick Start
Installation
# Clone the repository
git clone https://github.com/yourusername/marketing-econometrics.git
cd marketing-econometrics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Generate Synthetic Data
python data/data_generation.py
Run Notebooks
jupyter notebook
Navigate to notebooks/ and start with 01_adstock_and_saturation.ipynb.

Key Concepts Demonstrated
Adstock Transformation
Adstock captures the lagged and decaying effects of advertising:

from src.transformations import geometric_adstock

# Apply geometric adstock with 70% retention
adstocked_tv = geometric_adstock(tv_spend, decay_rate=0.7)
Saturation (Hill) Transformation
Models diminishing returns in marketing:

from src.transformations import hill_transform

# Apply Hill saturation
saturated_tv = hill_transform(adstocked_tv, alpha=50, k=2)
Marketing Mix Model
from src.models import BayesianMMM

model = BayesianMMM()
model.fit(X_transformed, y_sales)
contributions = model.get_channel_contributions()
Budget Optimization
from src.optimization import optimize_budget

optimal_allocation = optimize_budget(
    response_curves=model.response_curves,
    total_budget=1_000_000,
    bounds=channel_bounds
)
Example Outputs
Channel Attribution
Attribution Analysis

Response Curves
Response Curves

Optimization Results
Budget Optimization

Data Description
The synthetic dataset (data/synthetic_mmm_data.csv) includes:

52 weeks of marketing and sales data
Marketing channels: TV, Digital, Print, Radio
Control variables: Price, Promotions, Seasonality
Outcome: Sales revenue
Variables are generated to reflect realistic marketing dynamics including:

Adstock effects (carryover)
Saturation (diminishing returns)
Seasonality patterns
Noise and stochasticity
Technical Stack
Python 3.8+
PyMC: Bayesian modeling
scikit-learn: Machine learning and preprocessing
pandas: Data manipulation
numpy: Numerical computing
matplotlib/seaborn: Visualization
scipy: Optimization
Use Cases
This repository demonstrates solutions for:

Attribution analysis: Which channels drive the most sales?
Budget optimization: How should we allocate marketing spend?
Incrementality testing: What is the causal effect of each channel?
Response curve estimation: What are the diminishing returns?
Forecasting: What are expected sales under different scenarios?
Further Reading
Original Medium Article - Comprehensive overview of marketing econometrics
PyMC Documentation - Bayesian modeling in Python
Google's Lightweight MMM - Production-grade MMM
Meta's Robyn - MMM in R
Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

License
MIT License - feel free to use this code for your own projects.

Citation
If you use this code in your research or work, please cite:

@misc{marketing_econometrics_2025,
  title={Marketing Econometrics: A Practical Guide},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/marketing-econometrics}
}
Contact
For questions or feedback, please open an issue or reach out via [your contact method].

Disclaimer: This repository is for educational purposes. The synthetic data and models are simplified representations. Real-world applications require domain expertise, rigorous validation, and adaptation to specific business contexts.
