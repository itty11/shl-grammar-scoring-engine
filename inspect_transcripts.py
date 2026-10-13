
from pathlib import Path
import pandas as pd
import whisper

DATASET = Path(
    r"D:\AI Projects\shl-hiring-assessment-2026\Dataset_Final"
)

# Six examples: three zero-score and three nonzero-score
filenames = [
    "audio_5055.wav",
    "audio_5066.wav",
    "audio_5037.wav",
    "audio_126.wav",
    "audio_192.wav",
    "audio_447.wav",
]

train_df = pd.read_csv(DATASET / "train.csv")
labels = train_df.set_index("filename")["label"]

# English-only model; CPU-friendly starting point
model = whisper.load_model("base.en", device="cpu")

results = []

for filename in filenames:
    path = DATASET / "train" / filename
    print(f"\nTranscribing: {filename}")

    try:
        result = model.transcribe(
            str(path),
            fp16=False,
            language="en",
            verbose=False
        )

        transcript = result["text"].strip()

        print("Label:", labels[filename])
        print("Transcript:", transcript or "[No speech recognized]")

        results.append({
            "filename": filename,
            "label": labels[filename],
            "transcript": transcript
        })

    except Exception as exc:
        print("Error:", exc)

pd.DataFrame(results).to_csv(
    DATASET / "sample_transcripts.csv",
    index=False
)

print("\nSaved sample_transcripts.csv")
