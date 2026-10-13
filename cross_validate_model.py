import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from scipy.stats import pearsonr

# Reuse the preprocessing and dataset preparation
# from the existing multimodal baseline script.
exec(
    open("baseline_multimodal.py", encoding="utf-8")
    .read()
    .split("model = Pipeline([")[0]
)

# Use the same preprocessing pipeline, but evaluate it
# across five different train-validation folds.
cv_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def pearson_correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

scoring = {
    "rmse": make_scorer(
        root_mean_squared_error,
        greater_is_better=False
    ),
    "pearson": make_scorer(pearson_correlation)
}

results = cross_validate(
    cv_model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1
)

train_rmse = -results["train_rmse"]
val_rmse = -results["test_rmse"]

print("\n5-FOLD CROSS-VALIDATION: RIDGE alpha=1.0")
print("Training RMSE per fold:", np.round(train_rmse, 4))
print("Validation RMSE per fold:", np.round(val_rmse, 4))
print(f"Mean training RMSE:   {train_rmse.mean():.4f}")
print(f"Mean validation RMSE: {val_rmse.mean():.4f}")
print(f"Validation RMSE std:  {val_rmse.std():.4f}")
fold_pearson = results["test_pearson"]

print("Pearson correlation per fold:", np.round(fold_pearson, 4))
print(f"Mean validation Pearson: {fold_pearson.mean():.4f}")
print(f"Pearson std:              {fold_pearson.std():.4f}")

pd.DataFrame({
    "fold": np.arange(1, 6),
    "training_rmse": train_rmse,
    "validation_rmse": val_rmse
}).to_csv("cross_validation_results.csv", index=False)

print("\nSaved cross_validation_results.csv")