from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

train_df = pd.read_csv(DATASET / "train.csv")
results = []

for _, row in train_df.iterrows():
    path = DATASET / "train" / row["filename"]
    audio, sr = sf.read(path, dtype="float32")

    # Overall signal level
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

    # Fraction of samples with very low amplitude
    near_silence = float(np.mean(np.abs(audio) < 0.001))

    results.append({
        "filename": row["filename"],
        "label": row["label"],
        "duration_sec": len(audio) / sr,
        "rms": rms,
        "peak": peak,
        "near_silence_fraction": near_silence,
    })

df = pd.DataFrame(results)
df.to_csv(DATASET / "signal_features.csv", index=False)

print("Signal summary by label group:")
df["label_group"] = np.where(df["label"] == 0, "zero", "nonzero")

print(
    df.groupby("label_group")[
        ["duration_sec", "rms", "peak", "near_silence_fraction"]
    ].mean().round(4)
)

print("\nZero-score recordings:")
print(
    df[df["label"] == 0][
        ["filename", "duration_sec", "rms",
         "peak", "near_silence_fraction"]
    ].round(4).to_string(index=False)
)

print("\nSaved:", DATASET / "signal_features.csv")
