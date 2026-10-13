
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)
INPUT_CSV = DATASET / "train_transcripts.csv"

# 1. Load and validate the transcription dataset.
df = pd.read_csv(INPUT_CSV)

required = {"filename", "label", "transcript", "status"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[df["status"] == "ok"].copy()
df["transcript"] = df["transcript"].fillna("").astype(str)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"]).reset_index(drop=True)

if len(df) < 20:
    raise ValueError("Too few successfully transcribed recordings.")

X = df["transcript"]
y = df["label"].astype(float)

print("Usable recordings:", len(df))
print("Unique labels:", sorted(y.unique()))
print("Mean label:", round(y.mean(), 3))

# 2. Create a reproducible 80/20 split.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 3. Combine word-level and character-level TF-IDF features.
features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20000,
            sublinear_tf=True
        )
    )
])

X_train_features = features.fit_transform(X_train)
X_val_features = features.transform(X_val)

# 4. Compare against a simple mean-prediction baseline.
mean_prediction = np.full(len(y_val), y_train.mean())

mean_rmse = np.sqrt(
    mean_squared_error(y_val, mean_prediction)
)
mean_pearson = pearsonr(y_val, mean_prediction)[0]

print("\nMean baseline")
print(f"Validation RMSE: {mean_rmse:.4f}")
print(f"Validation Pearson: {mean_pearson}")

# 5. Fit Ridge regression on the training split.
model = Ridge(alpha=3.0)
model.fit(X_train_features, y_train)

train_predictions = model.predict(X_train_features)
val_predictions = model.predict(X_val_features)

# Grammar scores are bounded between 0 and 5.
train_predictions = np.clip(train_predictions, 0, 5)
val_predictions = np.clip(val_predictions, 0, 5)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)
val_rmse = np.sqrt(
    mean_squared_error(y_val, val_predictions)
)
val_pearson = pearsonr(y_val, val_predictions)[0]

print("\nTF-IDF + Ridge regression")
print(f"Training RMSE:   {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation Pearson: {val_pearson:.4f}")

# 6. Save predictions for error analysis.
predictions_df = pd.DataFrame({
    "filename": df.loc[X_val.index, "filename"].to_numpy(),
    "actual_score": y_val.to_numpy(),
    "predicted_score": val_predictions,
    "absolute_error": np.abs(
        y_val.to_numpy() - val_predictions
    )
})

predictions_df = predictions_df.sort_values(
    "absolute_error", ascending=False
)

predictions_df.to_csv(
    DATASET / "validation_predictions.csv",
    index=False
)

print("\nSaved validation_predictions.csv")
print("\nLargest prediction errors:")
print(predictions_df.head(10).to_string(index=False))
