# DC Rentability Analysis — Capstone Project (QSSR Track)

Author: Cameren Spicher  
Course: INST 414 – Capstone Project  
Institution: University of Maryland, College Park  
Track: Quantitative Social Science Research (QSSR)  
Semester: Fall 2025  

## Project Overview
This capstone analyzes rental profitability and neighborhood stability across Washington, DC census tracts. It explores where small-scale landlords can achieve sustainable and equitable returns while maintaining tenant stability and affordability. The project uses American Community Survey (ACS 5-Year DP04) data to evaluate rent levels, vacancy rates, turnover, overcrowding, and rent burden. Newly added Sprint 3 work expands the modeling to include a full OLS regression with robust errors, standardized coefficients, cross-validation performance, and a supplemental logistic model identifying high-vacancy tracts. Future work will integrate Redfin and Zillow ZORI data to estimate gross rental yield and construct a composite Rentability Score.

## Repository Structure
414_capstone/  
├─ README.md  
├─ data/  
│  ├─ raw/                          # source data (ACS DP04)  
│  └─ processed/                    # cleaned and engineered data  
├─ src/  
│  ├─ make_dataset.py               # data cleaning and feature creation  
│  ├─ models/  
│  │   └─ vacancy.py                # full Sprint 3 modeling pipeline  
│  └─ model_baseline.py             # baseline regression model (Sprint 2)  
├─ notebooks/  
│  └─ eda_visuals.py                # exploratory data visualizations  
├─ reports/  
│  └─ Sprint_2_Report.md            # earlier sprint report  
├─ figures/                         # generated plots (OLS + diagnostics)  
└─ outputs/                         # model results, coefficients, metrics  

## Setup
Clone the repo:
git clone https://github.com/CamerenSpicher/414_capstone.git  
cd 414_capstone

Install packages:
pip install -r requirements.txt  
or manually:
pip install pandas numpy matplotlib scikit-learn statsmodels

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

## Baseline Statistical Model (Sprint 2)
Run:
python src/model_baseline.py

Performs an initial OLS regression predicting rental vacancy rate using:
- Median rent  
- Renter share  
- Recent movers share  
- Overcrowded share  
- Median rooms  

Outputs coefficients to `outputs/baseline_linear_regression_coefficients.csv`.

## Sprint 3 Modeling Pipeline
Run:
python src/models/vacancy.py

This script performs the full modeling workflow:
- Loads cleaned ACS dataset  
- Creates train/test split for tract-level rental vacancy  
- Computes baseline (mean-prediction) MAE and RMSE  
- Fits OLS regression with heteroscedasticity-robust (HC3) standard errors  
- Evaluates train/test performance  
- Conducts 5-fold cross-validation  
- Computes standardized coefficients to rank predictor importance  
- Generates model diagnostics: residual plots, fitted–actual scatter, and rent-quantile error analysis  
- Adds a binary “high-vacancy” indicator (top 25% tracts)  
- Fits logistic regression to identify predictors of high vacancy  
- Saves all tables to `outputs/` and all plots to `figures/`

Key Sprint 3 outputs include:
- baseline_vacancy_metrics.csv  
- ols_vacancy_metrics.csv  
- ols_vacancy_cv_metrics.csv  
- ols_vacancy_standardized_coefficients.csv  
- logit_high_vacancy_metrics.csv  
- logit_high_vacancy_coefficients.csv  
- full OLS summary (`ols_vacancy_summary.txt`)  
- residual and diagnostic figures in `figures/`

## Analytical Objectives
Sprint 1: Proposal and data identification  
Sprint 2: Data acquisition, cleaning, and baseline regression  
Sprint 3: Full modeling pipeline (OLS, CV, standardized coefficients, logistic model)  
Sprint 4: Integrate Redfin/ZORI data, build Rentability Score, finalize report

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
Sprint 3 complete: full vacancy modeling workflow implemented, cross-validation and logistic model added  
Next: integrate Redfin/ZORI data and build Rentability Score  
Final: compile full report and presentation

## Ethical Considerations
All datasets are public and aggregated. No personal data used. Analyses focus on neighborhood-level patterns to support equitable housing insights.

## License
Developed for academic use at the University of Maryland. Code may be reused for educational or noncommercial purposes with attribution.
