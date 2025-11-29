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

# vacancy.py is in src/models, so parents[2] = repo root (414_capstone)
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

    # Drop rows missing outcome or key predictor
    df = df.dropna(subset=["vacancy_rate_rental", "median_rent"])
    return df


def train_test_split_data(df: pd.DataFrame):
    """Split into train and test sets for continuous vacancy outcome."""
    X = df[PREDICTOR_COLS]
    y = df["vacancy_rate_rental"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def compute_baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Baseline model that predicts the train mean vacancy for every tract."""
    baseline_value = y_train.mean()
    y_pred = np.repeat(baseline_value, len(y_test))
    mae = mean_absolute_error(y_test, y_pred)
    # Compatible with older sklearn: take square root manually
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    return {
        "baseline_mean": baseline_value,
        "mae": mae,
        "rmse": rmse,
    }


def fit_ols_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit OLS with robust standard errors."""
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit(cov_type="HC3")
    return model


def evaluate_ols_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
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


def cross_validate_ols(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Simple k-fold cross validation for OLS model.
    Returns a DataFrame with MAE and RMSE per fold.
    """
    X = df[PREDICTOR_COLS]
    y = df["vacancy_rate_rental"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    records = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = fit_ols_model(X_train, y_train)
        metrics, _, _, _, _ = evaluate_ols_model(
            model, X_train, y_train, X_test, y_test
        )

        records.append(
            {
                "fold": fold,
                "mae": metrics["test_mae"],
                "rmse": metrics["test_rmse"],
            }
        )

    return pd.DataFrame.from_records(records)


def compute_standardized_coefficients(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Compute standardized coefficients:
    beta_std_j = beta_j * (sd(X_j) / sd(y))
    """
    coef = model.params
    coef = coef.drop(labels=["const"])

    X_std = X_train[PREDICTOR_COLS].std()
    y_std = y_train.std()

    beta_std = coef * (X_std / y_std)

    coef_table = pd.DataFrame(
        {
            "variable": coef.index,
            "coef": coef.values,
            "std_coef": beta_std.values,
            "p_value": model.pvalues.drop(labels=["const"]).values,
        }
    )

    return coef_table.sort_values("std_coef", key=lambda s: s.abs(), ascending=False)


def add_high_vacancy_indicator(df: pd.DataFrame, quantile: float = 0.75):
    """
    Add a binary high_vacancy column where 1 indicates tracts at or above
    the chosen quantile of vacancy_rate_rental.
    """
    threshold = df["vacancy_rate_rental"].quantile(quantile)
    df = df.copy()
    df["high_vacancy"] = (df["vacancy_rate_rental"] >= threshold).astype(int)
    return df, threshold


def fit_logistic_model(df: pd.DataFrame):
    """
    Fit a logistic regression model predicting high_vacancy.
    Uses the same predictors as the OLS model.
    Returns model, metrics dict, coefficient table, probabilities, predictions,
    confusion matrix, and y_test for plotting.
    """
    X = df[PREDICTOR_COLS]
    y = df["high_vacancy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)

    prob_test = logit.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob_test)
    acc = accuracy_score(y_test, pred_test)
    cm = confusion_matrix(y_test, pred_test)

    metrics = {
        "auc": auc,
        "accuracy": acc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    coef_df = pd.DataFrame(
        {
            "variable": ["intercept"] + PREDICTOR_COLS,
            "coef": np.concatenate([[logit.intercept_[0]], logit.coef_[0]]),
        }
    )

    return logit, metrics, coef_df, prob_test, pred_test, cm, y_test


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_standardized_coefficients(coef_df: pd.DataFrame, out_path: pathlib.Path):
    """Bar plot of standardized coefficients."""
    plt.figure(figsize=(8, 5))
    ordered = coef_df.sort_values("std_coef", key=lambda s: s.abs(), ascending=True)
    plt.barh(ordered["variable"], ordered["std_coef"])
    plt.xlabel("Standardized coefficient")
    plt.title("Standardized coefficients for predictors of rental vacancy rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residuals_vs_fitted(
    fitted: pd.Series,
    residuals: pd.Series,
    out_path: pathlib.Path,
    title: str = "Residuals vs fitted values",
):
    plt.figure(figsize=(6, 5))
    plt.scatter(fitted, residuals, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Fitted vacancy rate")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_histogram(residuals: pd.Series, out_path: pathlib.Path):
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Distribution of OLS residuals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(
    y_test: pd.Series, y_pred: pd.Series, out_path: pathlib.Path
):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1)
    plt.xlabel("Actual vacancy rate")
    plt.ylabel("Predicted vacancy rate")
    plt.title("Actual vs predicted rental vacancy rates (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_by_rent_quantile(
    df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    out_path: pathlib.Path,
):
    """
    Plot mean absolute error by quartile of median_rent for the test set.
    """
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


def plot_logit_roc_curve(
    y_test: pd.Series,
    prob_test: np.ndarray,
    out_path: pathlib.Path,
):
    """Plot ROC curve for the high vacancy logistic classifier."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, prob_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve – high vacancy classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: pathlib.Path):
    """Plot confusion matrix for the high vacancy logistic classifier."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix – high vacancy classifier")
    plt.colorbar()

    classes = ["Not high vacancy", "High vacancy"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(ticks=tick_marks, labels=classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main run
# -----------------------------

def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} tracts from {DATA_PATH.name}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Baseline model
    baseline = compute_baseline_metrics(y_train, y_test)
    print("\nBaseline model:")
    print(baseline)

    pd.DataFrame([baseline]).to_csv(
        OUTPUTS_DIR / "baseline_vacancy_metrics.csv", index=False
    )

    # 4. OLS model
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

    # 5. Cross validation
    cv_results = cross_validate_ols(df)
    cv_results.to_csv(OUTPUTS_DIR / "ols_vacancy_cv_metrics.csv", index=False)
    print("\nCross validation results (MAE and RMSE per fold):")
    print(cv_results.describe()[["mae", "rmse"]])

    # 6. Standardized coefficients
    coef_df = compute_standardized_coefficients(ols_model, X_train, y_train)
    coef_df.to_csv(
        OUTPUTS_DIR / "ols_vacancy_standardized_coefficients.csv", index=False
    )

    # 7. Plots for report
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

    # 8. Logistic model for high vacancy (robustness)
    df_high, threshold = add_high_vacancy_indicator(df, quantile=0.75)
    print(f"\nHigh vacancy threshold (75th percentile): {threshold:.2f}")

    (
        logit_model,
        logit_metrics,
        logit_coef_df,
        prob_test,
        pred_test,
        cm,
        y_test_logit,
    ) = fit_logistic_model(df_high)

    print("\nLogistic model metrics (high vacancy vs not):")
    print(logit_metrics)

    logit_metrics["high_vacancy_threshold"] = threshold
    pd.DataFrame([logit_metrics]).to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_metrics.csv", index=False
    )
    logit_coef_df.to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_coefficients.csv", index=False
    )

    # Logistic model plots
    plot_logit_roc_curve(
        y_test_logit,
        prob_test,
        FIGURES_DIR / "sprint3_logit_roc_curve.png",
    )

    plot_confusion_matrix(
        cm,
        FIGURES_DIR / "sprint3_logit_confusion_matrix.png",
    )

    print("\nDone. Outputs written to:")
    print(f"  {OUTPUTS_DIR}")
    print(f"  {FIGURES_DIR}")


if __name__ == "__main__":
    main()


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_standardized_coefficients(coef_df: pd.DataFrame, out_path: pathlib.Path):
    """Bar plot of standardized coefficients."""
    plt.figure(figsize=(8, 5))
    ordered = coef_df.sort_values("std_coef", key=lambda s: s.abs(), ascending=True)
    plt.barh(ordered["variable"], ordered["std_coef"])
    plt.xlabel("Standardized coefficient")
    plt.title("Standardized coefficients for predictors of rental vacancy rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residuals_vs_fitted(
    fitted: pd.Series,
    residuals: pd.Series,
    out_path: pathlib.Path,
    title: str = "Residuals vs fitted values",
):
    plt.figure(figsize=(6, 5))
    plt.scatter(fitted, residuals, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Fitted vacancy rate")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_histogram(residuals: pd.Series, out_path: pathlib.Path):
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Distribution of OLS residuals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(
    y_test: pd.Series, y_pred: pd.Series, out_path: pathlib.Path
):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1)
    plt.xlabel("Actual vacancy rate")
    plt.ylabel("Predicted vacancy rate")
    plt.title("Actual vs predicted rental vacancy rates (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_by_rent_quantile(
    df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    out_path: pathlib.Path,
):
    """
    Plot mean absolute error by quartile of median_rent for the test set.
    """
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


def plot_logit_roc_curve(y_test: pd.Series, prob_test: np.ndarray, out_path: pathlib.Path):
    """Plot ROC curve for the high vacancy logistic classifier."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, prob_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve – high vacancy classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: pathlib.Path):
    """Plot confusion matrix for the high vacancy logistic classifier."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix – high vacancy classifier")
    plt.colorbar()

    classes = ["Not high vacancy", "High vacancy"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main run
# -----------------------------

def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} tracts from {DATA_PATH.name}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Baseline model
    baseline = compute_baseline_metrics(y_train, y_test)
    print("\nBaseline model:")
    print(baseline)

    pd.DataFrame([baseline]).to_csv(
        OUTPUTS_DIR / "baseline_vacancy_metrics.csv", index=False
    )

    # 4. OLS model
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

    # 5. Cross validation
    cv_results = cross_validate_ols(df)
    cv_results.to_csv(OUTPUTS_DIR / "ols_vacancy_cv_metrics.csv", index=False)
    print("\nCross validation results (MAE and RMSE per fold):")
    print(cv_results.describe()[["mae", "rmse"]])

    # 6. Standardized coefficients
    coef_df = compute_standardized_coefficients(ols_model, X_train, y_train)
    coef_df.to_csv(
        OUTPUTS_DIR / "ols_vacancy_standardized_coefficients.csv", index=False
    )

    # 7. Plots for report
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

    # 8. Logistic model for high vacancy (robustness)
    df_high, threshold = add_high_vacancy_indicator(df, quantile=0.75)
    print(f"\nHigh vacancy threshold (75th percentile): {threshold:.2f}")

    (
        logit_model,
        logit_metrics,
        logit_coef_df,
        prob_test,
        pred_test,
        cm,
        y_test_logit,
    ) = fit_logistic_model(df_high)

    print("\nLogistic model metrics (high vacancy vs not):")
    print(logit_metrics)

    logit_metrics["high_vacancy_threshold"] = threshold
    pd.DataFrame([logit_metrics]).to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_metrics.csv", index=False
    )
    logit_coef_df.to_csv(
        OUTPUTS_DIR / "logit_high_vacancy_coefficients.csv", index=False
    )

    # Logistic model plots
    plot_logit_roc_curve(
        y_test_logit,
        prob_test,
        FIGURES_DIR / "sprint3_logit_roc_curve.png",
    )

    plot_confusion_matrix(
        cm,
        FIGURES_DIR / "sprint3_logit_confusion_matrix.png",
    )

    print("\nDone. Outputs written to:")
    print(f"  {OUTPUTS_DIR}")
    print(f"  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
