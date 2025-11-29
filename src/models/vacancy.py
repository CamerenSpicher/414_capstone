"""
Sprint 3 modeling script for DC Rentability project (QSSR track).

This script:
- Loads the cleaned ACS tract level dataset
- Builds a baseline (mean) model for vacancy_rate_rental
- Fits an OLS regression model predicting vacancy_rate_rental
- Fits a robustness logistic model for "high vacancy" tracts
- Evaluates models with train/test split and cross validation
- Saves coefficient tables, metrics, and plots for the Sprint 3 report
"""

import pathlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -----------------------------
# Paths and constants
# -----------------------------

RANDOM_STATE = 42

# ROOT = repo root (414_capstone)
ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "dc_acs_cleaned_with_features.csv"
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"

FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

PREDICTOR_COLS = [
    "median_rent",
    "renter_share",
    "recent_movers_share",
    "overcrowded_share",
    "rent_burden35_share",
    "median_rooms",
]


# -----------------------------
# Helper functions
# -----------------------------

def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load cleaned ACS tract level data and drop missing key fields."""
    df = pd.read_csv(path)

    required = ["vacancy_rate_rental", "median_rent"] + PREDICTOR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=["vacancy_rate_rental", "median_rent"])
    return df


def train_test_split_data(df: pd.DataFrame):
    """Split into train and test sets for continuous vacancy outcome."""
    X = df[PREDICTOR_COLS]
    y = df["vacancy_rate_rental"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def compute_baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Baseline model predicting train-set mean vacancy for every tract."""
    baseline_value = y_train.mean()
    y_pred = np.repeat(baseline_value, len(y_test))
    mae = mean_absolute_error(y_test, y_pred)
    # manual RMSE instead of squared=False
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    return {"baseline_mean": baseline_value, "mae": mae, "rmse": rmse}


def fit_ols_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit OLS with robust standard errors."""
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit(cov_type="HC3")
    return model


def evaluate_ols_model(model, X_train, y_train, X_test, y_test):
    """Evaluate OLS model on train and test sets and compute residuals."""
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    y_train_pred = model.predict(X_train_const)
    y_test_pred = model.predict(X_test_const)

    metrics = {
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_rmse": mean_squared_error(y_train, y_train_pred) ** 0.5,
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_rmse": mean_squared_error(y_test, y_test_pred) ** 0.5,
    }

    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    return metrics, y_train_pred, y_test_pred, residuals_train, residuals_test


def cross_validate_ols(df: pd.DataFrame, n_splits=5):
    """Simple k-fold CV for OLS model."""
    X = df[PREDICTOR_COLS]
    y = df["vacancy_rate_rental"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = fit_ols_model(X_train, y_train)
        metrics, _, y_test_pred, _, _ = evaluate_ols_model(
            model, X_train, y_train, X_test, y_test
        )

        records.append(
            {
                "fold": fold,
                "mae": metrics["test_mae"],
                "rmse": metrics["test_rmse"],
            }
        )

    return pd.DataFrame(records)


def compute_standardized_coefficients(model, X_train, y_train):
    """Compute standardized coefficients for interpretability."""
    coef = model.params.drop("const")
    X_std = X_train.std()
    y_std = y_train.std()
    beta_std = coef * (X_std / y_std)

    return (
        pd.DataFrame(
            {
                "variable": coef.index,
                "coef": coef.values,
                "std_coef": beta_std.values,
                "p_value": model.pvalues.drop("const").values,
            }
        )
        .sort_values("std_coef", key=lambda s: s.abs(), ascending=False)
    )


def add_high_vacancy_indicator(df, quantile=0.75):
    """Binary high vacancy indicator."""
    threshold = df["vacancy_rate_rental"].quantile(quantile)
    df2 = df.copy()
    df2["high_vacancy"] = (df2["vacancy_rate_rental"] >= threshold).astype(int)
    return df2, threshold


def fit_logistic_model(df):
    """Fit logistic regression on high vacancy indicator."""
    X = df[PREDICTOR_COLS]
    y = df["high_vacancy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)

    prob_test = logit.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob_test)
    acc = accuracy_score(y_test, pred_test)
    cm = confusion_matrix(y_test, pred_test)

    coef_df = pd.DataFrame(
        {
            "variable": ["intercept"] + PREDICTOR_COLS,
            "coef": np.concatenate([[logit.intercept_[0]], logit.coef_[0]]),
        }
    )

    metrics = {
        "auc": auc,
        "accuracy": acc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return logit, metrics, coef_df


# -----------------------------
# Plot helpers
# -----------------------------

def plot_standardized_coefficients(coef_df, out_path):
    plt.figure(figsize=(8, 5))
    ordered = coef_df.sort_values("std_coef", key=lambda s: s.abs(), ascending=True)
    plt.barh(ordered["variable"], ordered["std_coef"])
    plt.xlabel("Standardized coefficient")
    plt.title("Standardized coefficients for predictors of rental vacancy rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residuals_vs_fitted(fitted, residuals, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(fitted, residuals, alpha=0.6)
    plt.axhline(0, color="black")
    plt.xlabel("Fitted vacancy rate")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_histogram(residuals, out_path):
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Distribution of OLS residuals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, out_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red")
    plt.xlabel("Actual vacancy rate")
    plt.ylabel("Predicted vacancy rate")
    plt.title("Actual vs predicted rental vacancy rates (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_by_rent_quantile(df, y_test, y_pred, out_path):
    test_df = df.loc[y_test.index].copy()
    test_df["abs_error"] = (y_test - y_pred).abs()
    test_df["rent_quantile"] = pd.qcut(
        test_df["median_rent"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    grouped = test_df.groupby("rent_quantile")["abs_error"].mean()

    plt.figure(figsize=(6, 5))
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.xlabel("Median rent quartile")
    plt.ylabel("Mean absolute error")
    plt.title("Prediction error by neighborhood rent level (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    # Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} tracts from {DATA_PATH.name}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Baseline
    baseline = compute_baseline_metrics(y_train, y_test)
    print("\nBaseline model:")
    print(baseline)
    pd.DataFrame([baseline]).to_csv(
        OUTPUTS_DIR / "baseline_vacancy_metrics.csv", index=False
    )

    # OLS model
    ols_model = fit_ols_model(X_train, y_train)
    print("\nOLS model summary:")
    with open(OUTPUTS_DIR / "ols_vacancy_summary.txt", "w") as f:
        f.write(ols_model.summary().as_text())

    metrics, y_train_pred, y_test_pred, resid_train, resid_test = evaluate_ols_model(
        ols_model, X_train, y_train, X_test, y_test
    )
    print("\nOLS train/test metrics:")
    print(metrics)
    pd.DataFrame([metrics]).to_csv(
        OUTPUTS_DIR / "ols_vacancy_metrics.csv", index=False
    )

    # Cross validation
    cv_results = cross_validate_ols(df)
    cv_results.to_csv(OUTPUTS_DIR / "ols_vacancy_cv_metrics.csv", index=False)
    print("\nCross validation results (MAE and RMSE per fold):")
    print(cv_results.describe()[["mae", "rmse"]])

    # Standardized coefficients
    coef_df = compute_standardized_coefficients(ols_model, X_train, y_train)
    coef_df.to_csv(
        OUTPUTS_DIR / "ols_vacancy_standardized_coefficients.csv", index=False
    )

    # Plots
    plot_standardized_coefficients(
        coef_df, FIGURES_DIR / "sprint3_standardized_coefficients.png"
    )
    plot_residuals_vs_fitted(
        y_train_pred,
        resid_train,
        FIGURES_DIR / "sprint3_residuals_vs_fitted_train.png",
        title="Residuals vs fitted values (train set)",
    )
    plot_residual_histogram(
        resid_train, FIGURES_DIR / "sprint3_residual_histogram_train.png"
    )
    plot_actual_vs_predicted(
        y_test, y_test_pred, FIGURES_DIR / "sprint3_actual_vs_predicted_test.png"
    )
    plot_error_by_rent_quantile(
        df,
        y_test,
        y_test_pred,
        FIGURES_DIR / "sprint3_error_by_rent_quartile_test.png",
    )

    # Logistic model
    df_high, threshold = add_high_vacancy_indicator(df, 0.75)
    print(f"\nHigh vacancy threshold (75th percentile): {threshold:.2f}")

    logit_model, logit_metrics, logit_coef_df = fit_logistic_model(df_high)
    print("\nLogistic model metrics (high vacancy vs not):")
    print(logit_metrics)

    logit_metrics["high_vacancy_threshold"] = threshold
    pd.DataFrame([logit_metrics]).to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_metrics.csv", index=False
    )
    logit_coef_df.to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_coefficients.csv", index=False
    )

    print("\nDone. Outputs written to:")
    print(f"  {OUTPUTS_DIR}")
    print(f"  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
