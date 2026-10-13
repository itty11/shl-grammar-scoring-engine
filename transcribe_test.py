
from pathlib import Path

import pandas as pd
import whisper

# Project paths
DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)
TEST_DIR = DATASET / "test"
TEST_CSV = DATASET / "test.csv"
OUTPUT_CSV = DATASET / "test_transcripts.csv"

test_df = pd.read_csv(TEST_CSV)

# Load the same Whisper model used for training transcripts.
model = whisper.load_model("base.en", device="cpu")

# Resume safely if interrupted.
if OUTPUT_CSV.exists():
    results_df = pd.read_csv(OUTPUT_CSV)
    results = results_df.to_dict("records")

    completed = set(
        results_df.loc[
            results_df["status"].astype(str).eq("ok"),
            "filename"
        ].astype(str)
    )
else:
    results = []
    completed = set()

total = len(test_df)

for _, row in test_df.iterrows():
    filename = str(row["filename"])

    if filename in completed:
        continue

    audio_path = TEST_DIR / filename

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        continue

    try:
        result = model.transcribe(
            str(audio_path),
            language="en",
            fp16=False,
            verbose=False
        )

        segments = result.get("segments", [])

        avg_logprob = (
            sum(s.get("avg_logprob", 0.0) for s in segments)
            / len(segments)
            if segments else None
        )

        avg_no_speech = (
            sum(s.get("no_speech_prob", 0.0) for s in segments)
            / len(segments)
            if segments else None
        )

        record = {
            "filename": filename,
            "transcript": result["text"].strip(),
            "avg_logprob": avg_logprob,
            "avg_no_speech_prob": avg_no_speech,
            "status": "ok"
        }

    except Exception as exc:
        print(f"\nERROR: {filename}: {exc}")

        record = {
            "filename": filename,
            "transcript": "",
            "avg_logprob": None,
            "avg_no_speech_prob": None,
            "status": f"error: {exc}"
        }

    results = [
        r for r in results if str(r["filename"]) != filename
    ]
    results.append(record)

    # Save after each recording so work can be resumed.
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if record["status"] == "ok":
        completed.add(filename)

    print(f"[{len(completed)}/{total}] {filename}")

print(f"\nSaved transcripts to: {OUTPUT_CSV}")
print(f"Successfully transcribed: {len(completed)}/{total}")
