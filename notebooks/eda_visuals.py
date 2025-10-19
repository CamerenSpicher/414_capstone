#Sprint-2 visuals

import os
import pandas as pd
import matplotlib.pyplot as plt

PROC = "data/processed/dc_acs_cleaned_with_features.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(PROC)

def save_hist(series, title, xlabel, filename):
    s = series.dropna()
    plt.figure()
    plt.hist(s, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()

def save_scatter(x, y, title, xlabel, ylabel, filename):
    mask = x.notna() & y.notna()
    plt.figure()
    plt.scatter(x[mask], y[mask])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()

# ---- Univariate
save_hist(df["median_rent"], "Distribution of Median Gross Rent", "Median Gross Rent ($)", "hist_median_rent.png")
save_hist(df["vacancy_rate_rental"], "Distribution of Rental Vacancy Rate", "Rental Vacancy Rate (%)", "hist_vacancy_rate.png")
save_hist(df["rent_burden35_share"], "Distribution of Rent Burden (≥35% of Income)", "Share of Renters ≥35%", "hist_rent_burden.png")

# ---- Bivariate
save_scatter(df["renter_share"], df["median_rent"], "Renter Share vs Median Rent", "Renter Share (proportion)", "Median Gross Rent ($)", "scatter_renter_vs_rent.png")
save_scatter(df["renter_share"], df["vacancy_rate_rental"], "Renter Share vs Rental Vacancy Rate", "Renter Share (proportion)", "Rental Vacancy Rate (%)", "scatter_renter_vs_vacancy.png")

# ---- Additional Bivariate: Median Rent vs Vacancy Rate
save_scatter(
    df["median_rent"],
    df["vacancy_rate_rental"],
    "Median Gross Rent vs Rental Vacancy Rate",
    "Median Gross Rent ($)",
    "Rental Vacancy Rate (%)",
    "scatter_rent_vs_vacancy.png"
)

print("[eda_visuals] saved figures to:", FIG_DIR)
