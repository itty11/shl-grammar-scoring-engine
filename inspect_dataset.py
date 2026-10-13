from pathlib import Path
import pandas as pd

DATASET = Path(r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final")

# Inspect CSV files
for filename in ["train.csv", "test.csv", "sample_submission.csv"]:
    path = DATASET / filename
    df = pd.read_csv(path)

    print(f"\n{'=' * 45}")
    print(f"{filename}: {df.shape}")
    print("Columns:", df.columns.tolist())
    print(df.head(5).to_string(index=False))
    print("Missing values:\n", df.isnull().sum())

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False).sort_index())

# Inspect audio folders
for folder_name in ["train", "test"]:
    folder = DATASET / folder_name

    print(f"\n{'=' * 45}")
    print(f"Folder: {folder}")

    if folder.exists():
        files = list(folder.rglob("*.wav"))
        print("WAV count:", len(files))
        print("First 10 files:")
        for file in files[:10]:
            print(file.relative_to(DATASET))
    else:
        print("Folder not found")