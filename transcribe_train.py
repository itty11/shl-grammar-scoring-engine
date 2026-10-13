
from pathlib import Path

import pandas as pd
import whisper

DATASET = Path(r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final")
TRAIN_DIR = DATASET / "train"
TRAIN_CSV = DATASET / "train.csv"
OUTPUT_CSV = DATASET / "train_transcripts.csv"

train_df = pd.read_csv(TRAIN_CSV)
model = whisper.load_model("base.en", device="cpu")

# Resume from the previous run if the output file already exists.
if OUTPUT_CSV.exists():
    results_df = pd.read_csv(OUTPUT_CSV)
    completed = set(results_df["filename"].astype(str))
    results = results_df.to_dict("records")
else:
    completed = set()
    results = []

total = len(train_df)

for i, row in train_df.iterrows():
    filename = str(row["filename"])

    if filename in completed:
        continue

    audio_path = TRAIN_DIR / filename

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

        results.append({
            "filename": filename,
            "label": float(row["label"]),
            "transcript": result["text"].strip(),
            "avg_logprob": avg_logprob,
            "avg_no_speech_prob": avg_no_speech,
            "status": "ok"
        })

    except Exception as exc:
        print(f"\nERROR: {filename}: {exc}")

        results.append({
            "filename": filename,
            "label": float(row["label"]),
            "transcript": "",
            "avg_logprob": None,
            "avg_no_speech_prob": None,
            "status": f"error: {exc}"
        })

    completed.add(filename)

    # Save after each recording so interrupted work can be resumed.
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print(
        f"[{len(completed)}/{total}] "
        f"{filename} | label={row['label']}"
    )

print(f"\nFinished. Results saved to: {OUTPUT_CSV}")
print(f"Processed recordings: {len(completed)}/{total}")
