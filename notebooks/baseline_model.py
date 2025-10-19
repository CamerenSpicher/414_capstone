#predict rental vacancy rate

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PROC = "data/processed/dc_acs_cleaned_with_features.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PROC)

model_df = df[[
    "vacancy_rate_rental","median_rent","renter_share","recent_movers_share","overcrowded_share","median_rooms"
]].dropna()

X = model_df[["median_rent","renter_share","recent_movers_share","overcrowded_share","median_rooms"]]
y = model_df["vacancy_rate_rental"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)

coef = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
coef.loc[len(coef)] = {"feature": "intercept", "coefficient": model.intercept_}
coef_path = os.path.join(OUT_DIR, "baseline_linear_regression_coefficients.csv")
coef.to_csv(coef_path, index=False)

print("[baseline_model] Validation MAE (percentage points):", round(mae, 3))
print("[baseline_model] Coefficients saved to:", coef_path)
print(coef)
