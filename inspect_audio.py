from pathlib import Path
import pandas as pd
import soundfile as sf

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

train_df = pd.read_csv(DATASET / "train.csv")
results = []

# Inspect all training audio files
for _, row in train_df.iterrows():
    audio_path = DATASET / "train" / row["filename"]

    try:
        info = sf.info(audio_path)

        results.append({
            "filename": row["filename"],
            "label": row["label"],
            "duration_sec": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "error": None
        })

    except Exception as e:
        results.append({
            "filename": row["filename"],
            "label": row["label"],
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "error": str(e)
        })

df = pd.DataFrame(results)
df.to_csv(DATASET / "audio_metadata.csv", index=False)

print("Total files inspected:", len(df))
print("Files with errors:", df["error"].notna().sum())

print("\nAudio duration statistics:")
print(df["duration_sec"].describe())

print("\nSample rates:")
print(df["sample_rate"].value_counts(dropna=False))

print("\nChannels:")
print(df["channels"].value_counts(dropna=False))

print("\nZero-score audio metadata:")
print(
    df.loc[df["label"] == 0, [
        "filename", "duration_sec",
        "sample_rate", "channels", "error"
    ]].to_string(index=False)
)

print("\nSaved metadata to:", DATASET / "audio_metadata.csv")
