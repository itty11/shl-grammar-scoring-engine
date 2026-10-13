
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

test_df = pd.read_csv(DATASET / "test.csv")
results = []

for i, row in test_df.iterrows():
    path = DATASET / "test" / row["filename"]

    audio, sr = sf.read(path, dtype="float32")

    # Use exactly the same calculations as the training features
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    near_silence = float(np.mean(np.abs(audio) < 0.001))

    results.append({
        "filename": row["filename"],
        "duration_sec": len(audio) / sr,
        "rms": rms,
        "peak": peak,
        "near_silence_fraction": near_silence,
    })

    if (i + 1) % 25 == 0 or (i + 1) == len(test_df):
        print(f"Processed {i + 1}/{len(test_df)} test recordings")

df = pd.DataFrame(results)

output_path = DATASET / "test_signal_features.csv"
df.to_csv(output_path, index=False)

print("\nRows:", len(df))
print("Duplicate filenames:", df["filename"].duplicated().sum())
print("Missing values:\n", df.isnull().sum())
print("Saved:", output_path)
print("\nFeature summary:")
print(
    df[
        ["duration_sec", "rms", "peak", "near_silence_fraction"]
    ].describe().round(4)
)
