
from pathlib import Path
import pandas as pd

DATASET = Path(r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final")

train_df = pd.read_csv(DATASET / "train.csv")
test_df = pd.read_csv(DATASET / "test.csv")
sample_df = pd.read_csv(DATASET / "sample_submission.csv")

train_wavs = {p.name for p in (DATASET / "train").glob("*.wav")}
test_wavs = {p.name for p in (DATASET / "test").glob("*.wav")}

train_csv = set(train_df["filename"])
test_csv = set(test_df["filename"])
sample_csv = set(sample_df["filename"])

checks = {
    "Train CSV filenames missing from train folder":
        train_csv - train_wavs,
    "Train folder files missing from train CSV":
        train_wavs - train_csv,
    "Test CSV filenames missing from test folder":
        test_csv - test_wavs,
    "Test folder files missing from test CSV":
        test_wavs - test_csv,
    "Sample filenames missing from test folder":
        sample_csv - test_wavs,
    "Sample filenames missing from train folder":
        sample_csv - train_wavs,
}

for title, filenames in checks.items():
    print(f"\n{title}: {len(filenames)}")
    print(sorted(filenames)[:10])

print("\n--- OVERLAP COUNTS ---")
print("Train CSV ∩ Test CSV:", len(train_csv & test_csv))
print("Train CSV ∩ Sample:", len(train_csv & sample_csv))
print("Test CSV ∩ Sample:", len(test_csv & sample_csv))

print("\n--- ZERO-SCORE EXAMPLES ---")
print(
    train_df.loc[train_df["label"] == 0, "filename"]
    .head(10).to_list()
)
