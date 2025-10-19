```markdown
# DC Rentability Analysis — Capstone Project (QSSR Track)

Author: Cameren Spicher  
Course: INST 414 – Capstone Project  
Institution: University of Maryland, College Park  
Track: Quantitative Social Science Research (QSSR)  
Semester: Fall 2025  

## Project Overview
This capstone analyzes rental profitability and neighborhood stability across Washington, DC census tracts. It explores where small-scale landlords can achieve sustainable and equitable returns while maintaining tenant stability and affordability. The project uses American Community Survey (ACS 5-Year DP04) data to evaluate rent levels, vacancy rates, turnover, overcrowding, and rent burden. Future work will integrate Redfin and Zillow ZORI data to estimate gross rental yield and construct a composite Rentability Score.

## Repository Structure
414_capstone/
├─ README.md
├─ data/
│  ├─ raw/                          # source data (ACS DP04)
│  └─ processed/                    # cleaned and engineered data
├─ src/
│  ├─ make_dataset.py               # data cleaning and feature creation
│  └─ model_baseline.py             # baseline regression model
├─ notebooks/
│  └─ eda_visuals.py                # exploratory data visualizations
├─ reports/
│  └─ Sprint_2_Report.md            # sprint report
├─ figures/                         # generated plots
└─ outputs/                         # model results and summaries

## Setup
Clone the repo:
git clone https://github.com/CamerenSpicher/414_capstone.git
cd 414_capstone

Install packages:
pip install -r requirements.txt
or manually:
pip install pandas numpy matplotlib scikit-learn

## Data Sources
- DC Open Data: ACS 5-Year Housing Characteristics (primary dataset)
- Redfin Data Center (home sale prices)
- Zillow Observed Rent Index (ZORI)
- Open Data DC – Crime Incidents
- WMATA GTFS Stops (transit access)

## Data Preparation
Run cleaning script:
python src/make_dataset.py

This script loads and cleans ACS data, creates ratio-based features, standardizes geography, exports cleaned data to `data/processed/dc_acs_cleaned_with_features.csv`, and saves summary outputs.

## Exploratory Data Analysis
Run visualization script:
python notebooks/eda_visuals.py

Generates plots for:
- Median rent distribution
- Vacancy rate distribution
- Rent burden (≥35% of income)
- Renter share vs median rent
- Renter share vs vacancy rate

Saved in `figures/`.

## Baseline Statistical Model
Run:
python src/model_baseline.py

Performs OLS regression predicting rental vacancy rate using:
- Median rent
- Renter share
- Recent movers share
- Overcrowded share
- Median rooms

Outputs model coefficients to `outputs/baseline_linear_regression_coefficients.csv` and prints validation MAE.

## Analytical Objectives
Sprint 1: Proposal and data identification  
Sprint 2: Data acquisition, cleaning, and EDA  
Sprint 3: Integrate Redfin/ZORI data, build Rentability Score  
Sprint 4: Final report and presentation

## Key Variables
median_rent – Median gross rent (DP04_0134E)  
vacancy_rate_rental – Rental vacancy rate (DP04_0005E)  
renter_share – Renter-occupied share (DP04_0047E / DP04_0045E)  
recent_movers_share – Moved in 2021+ (DP04_0051E / DP04_0050E)  
overcrowded_share – >1.5 persons per room (DP04_0079E / DP04_0076E)  
rent_burden35_share – Renters paying ≥35% income (DP04_0142E / DP04_0136E)  
median_rooms – Median rooms per housing unit (DP04_0037E)

## Progress
Sprint 2 complete: dataset cleaned, EDA and baseline regression done  
Next: integrate Redfin/ZORI data and build Rentability Score  
Final: compile full report and presentation

## Ethical Considerations
All datasets are public and aggregated. No personal data used. Analyses focus on neighborhood-level patterns to support equitable housing insights.

## License
Developed for academic use at the University of Maryland. Code may be reused for educational or noncommercial purposes with attribution.
```
