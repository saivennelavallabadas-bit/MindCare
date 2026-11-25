from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/participant_manifest.csv"
OUTPUT = ROOT / "data/processed/facial_landmark_rotation_v1.csv"
ERRORS = ROOT / "reports/tables/facial_landmark_rotation_errors_v1.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ERRORS.parent.mkdir(parents=True, exist_ok=True)


def extract_landmark_features(file_path):
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

    x_cols = [f"x{i}" for i in range(68)]
    y_cols = [f"y{i}" for i in range(68)]

    valid = valid.dropna(subset=x_cols + y_cols)

    if len(valid) == 0:
        raise ValueError("No complete landmark frames")

    x = valid[x_cols].to_numpy(dtype=float)
    y = valid[y_cols].to_numpy(dtype=float)

    left_eye_x = x[:, 36:42].mean(axis=1)
    left_eye_y = y[:, 36:42].mean(axis=1)

    right_eye_x = x[:, 42:48].mean(axis=1)
    right_eye_y = y[:, 42:48].mean(axis=1)

    center_x = (left_eye_x + right_eye_x) / 2
    center_y = (left_eye_y + right_eye_y) / 2

    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y

    eye_distance = np.sqrt(dx ** 2 + dy ** 2)
    eye_angle = np.arctan2(dy, dx)

    valid_scale = eye_distance > 1e-6

    x = x[valid_scale]
    y = y[valid_scale]
    center_x = center_x[valid_scale]
    center_y = center_y[valid_scale]
    eye_distance = eye_distance[valid_scale]
    eye_angle = eye_angle[valid_scale]
    valid = valid.iloc[np.where(valid_scale)[0]]

    if len(x) == 0:
        raise ValueError("Invalid eye-distance normalization")

    x_centered = x - center_x[:, None]
    y_centered = y - center_y[:, None]

    cos_angle = np.cos(eye_angle)[:, None]
    sin_angle = np.sin(eye_angle)[:, None]

    x_rotated = x_centered * cos_angle + y_centered * sin_angle
    y_rotated = -x_centered * sin_angle + y_centered * cos_angle

    x_normalized = x_rotated / eye_distance[:, None]
    y_normalized = y_rotated / eye_distance[:, None]

    features = {}

    features["total_frames"] = total_frames
    features["valid_frames"] = len(valid)
    features["valid_frame_ratio"] = len(valid) / total_frames
    features["confidence_mean"] = valid["confidence"].mean()
    features["eye_distance_mean"] = eye_distance.mean()
    features["eye_distance_std"] = eye_distance.std()
    features["eye_angle_mean"] = eye_angle.mean()
    features["eye_angle_std"] = eye_angle.std()

    for i in range(68):
        features[f"landmark_x{i}_mean"] = x_normalized[:, i].mean()
        features[f"landmark_x{i}_std"] = x_normalized[:, i].std()

        features[f"landmark_y{i}_mean"] = y_normalized[:, i].mean()
        features[f"landmark_y{i}_std"] = y_normalized[:, i].std()

    return features


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["facial_landmarks_available"] == True].copy()

    results = []
    errors = []

    total = len(df)

    print("=" * 65)
    print("FACIAL LANDMARK ROTATION NORMALIZATION V1")
    print("=" * 65)
    print("Participants:", total)

    for _, row in df.iterrows():
        participant_id = row["participant_id"]
        file_path = row["facial_landmarks_path"]

        try:
            features = extract_landmark_features(file_path)

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
    print("EXPERIMENT FINISHED")
    print("=" * 65)
    print("Successful:", len(results))
    print("Failed:", len(errors))
    print("Feature table shape:", feature_df.shape)
    print("Features saved:", OUTPUT)
    print("Errors saved:", ERRORS)


if __name__ == "__main__":
    main()