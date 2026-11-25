
from pathlib import Path
import pandas as pd

ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/participant_manifest.csv"
OUTPUT = ROOT / "data/processed/text_documents.csv"

manifest = pd.read_csv(MANIFEST)
records = []

for _, row in manifest.iterrows():
    participant_id = row["participant_id"]
    split = row["split"]
    score = row["phq8_score"]
    transcript_path = row["transcript_path"]

    document = ""

    if pd.notna(transcript_path) and Path(str(transcript_path)).exists():
        transcript = pd.read_csv(transcript_path, sep="\t")

        participant_rows = transcript[
            transcript["speaker"].astype(str).str.strip().str.lower() == "participant"
        ]

        utterances = participant_rows["value"].dropna().astype(str).tolist()
        document = " ".join(utterances)

    records.append({
        "participant_id": participant_id,
        "split": split,
        "phq8_score": score,
        "participant_text": document
    })

output = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT, index=False, encoding="utf-8")

print("=" * 60)
print("TEXT DOCUMENT EXTRACTION")
print("=" * 60)
print("Participants:", len(output))
print("Non-empty documents:", (output["participant_text"].str.strip() != "").sum())
print("Empty documents:", (output["participant_text"].str.strip() == "").sum())
print("Output:", OUTPUT)
print("=" * 60)
