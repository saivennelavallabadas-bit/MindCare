from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/participant_manifest.csv"
OUTPUT = ROOT / "data/processed/facial_au_features.csv"
ERRORS = ROOT / "reports/tables/facial_au_extraction_errors.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ERRORS.parent.mkdir(parents=True, exist_ok=True)


def extract_facial_au_features(file_path):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    df = df.apply(pd.to_numeric, errors="coerce")

    total_frames = len(df)

    df = df.dropna(subset=["confidence", "success"])

    valid = df[
        (df["success"] == 1) &
        (df["confidence"] >= 0.80)
    ].copy()

    if len(valid) == 0:
        raise ValueError("No frames passed the quality filter")

    features = {}

    features["total_frames"] = total_frames
    features["valid_frames"] = len(valid)
    features["valid_frame_ratio"] = len(valid) / total_frames
    features["confidence_mean"] = df["confidence"].mean()
    features["valid_confidence_mean"] = valid["confidence"].mean()

    au_r = [
        c for c in valid.columns
        if c.endswith("_r") and c.startswith("AU")
    ]

    au_c = [
        c for c in valid.columns
        if c.endswith("_c") and c.startswith("AU")
    ]

    for col in au_r:
        features[f"{col}_mean"] = valid[col].mean()
        features[f"{col}_std"] = valid[col].std()

    for col in au_c:
        features[f"{col}_mean"] = valid[col].mean()
        features[f"{col}_std"] = valid[col].std()

    return features


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["facial_aus_available"] == True].copy()

    results = []
    errors = []

    total = len(df)

    print("=" * 65)
    print("FACIAL ACTION UNIT FEATURE EXTRACTION")
    print("=" * 65)
    print("Participants:", total)

    for _, row in df.iterrows():
        participant_id = row["participant_id"]
        file_path = row["facial_aus_path"]

        try:
            features = extract_facial_au_features(file_path)

            results.append({
                "participant_id": participant_id,
                "split": row["split"],
                **features
            })

        except Exception as e:
            errors.append({
                "participant_id": participant_id,
                "file_path": file_path,
                "error": str(e)
            })

        completed = len(results) + len(errors)

        if completed % 10 == 0 or completed == total:
            print(
                f"Progress: {completed}/{total} | "
                f"Success: {len(results)} | Errors: {len(errors)}"
            )

    feature_df = pd.DataFrame(results)
    feature_df.to_csv(OUTPUT, index=False)

    error_df = pd.DataFrame(errors)
    error_df.to_csv(ERRORS, index=False)

    print("\n" + "=" * 65)
    print("FACIAL AU EXTRACTION FINISHED")
    print("=" * 65)
    print("Successful:", len(results))
    print("Failed:", len(errors))
    print("Feature table shape:", feature_df.shape)
    print("Features saved:", OUTPUT)
    print("Errors saved:", ERRORS)


if __name__ == "__main__":
    main()