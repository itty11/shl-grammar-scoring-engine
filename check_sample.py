import pandas as pd
from pathlib import Path

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

sample = pd.read_csv(DATASET / "sample_submission.csv")
test = pd.read_csv(DATASET / "test.csv")

print("Sample submission rows:", len(sample))
print("Test CSV rows:", len(test))

print("\nSample submission filenames:")
print(sample["filename"].head(10).tolist())

print("\nTest CSV filenames:")
print(test["filename"].head(10).tolist())
