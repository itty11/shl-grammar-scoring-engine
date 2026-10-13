
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

# Load existing transcripts and audio-signal features.
df = pd.read_csv(DATASET / "train_transcripts.csv")
audio = pd.read_csv(DATASET / "signal_features.csv")

df = df[df["status"] == "ok"].copy()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])

df["transcript"] = df["transcript"].fillna("").astype(str)

# Join by filename, never by row order.
audio = audio.drop(
    columns=["label"], errors="ignore"
)
df = df.merge(
    audio,
    on="filename",
    how="left",
    validate="one_to_one",
    suffixes=("", "_audio")
)

# Numeric audio and Whisper diagnostic features.
excluded = {
    "label", "filename", "transcript", "status"
}
numeric_cols = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in excluded
]

if not numeric_cols:
    raise ValueError(
        "No numeric audio/Whisper features found. "
        "Check signal_features.csv and train_transcripts.csv."
    )

# Remove infinite values; imputation happens inside the pipeline.
df[numeric_cols] = df[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)

print("Recordings:", len(df))
print("Numeric features:", numeric_cols)
print("Missing values per numeric feature:")
print(df[numeric_cols].isna().sum().to_string())

X = df[["transcript"] + numeric_cols]
y = df["label"].astype(float)

# Same reproducible 80/20 split as the text-only baseline.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

text_features = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    ))
])

numeric_features = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("text", text_features, "transcript"),
    ("audio", numeric_features, numeric_cols)
])

model = Pipeline([
    ("features", preprocessor),
    ("regressor", Ridge(alpha=3.0))
])

model.fit(X_train, y_train)

train_pred = np.clip(model.predict(X_train), 0, 5)
val_pred = np.clip(model.predict(X_val), 0, 5)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_pred)
)
val_rmse = np.sqrt(
    mean_squared_error(y_val, val_pred)
)
val_pearson = pearsonr(y_val, val_pred)[0]

print("\nTEXT + AUDIO BASELINE")
print(f"Training RMSE:   {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation Pearson: {val_pearson:.4f}")

# Save predictions and identify the remaining errors.
results = pd.DataFrame({
    "filename": X_val.index.map(df["filename"]),
    "actual_score": y_val.to_numpy(),
    "predicted_score": val_pred,
})
results["absolute_error"] = np.abs(
    results["actual_score"] - results["predicted_score"]
)

results.to_csv(
    DATASET / "validation_multimodal_predictions.csv",
    index=False
)

print("\nLargest errors:")
print(
    results.sort_values("absolute_error", ascending=False)
    .head(10)
    .to_string(index=False)
)
