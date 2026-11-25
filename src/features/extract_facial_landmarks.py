
import pandas as pd
import numpy as np


def extract_landmark_features(file_path):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    df.columns = [
        c.replace("x_", "x", 1) if c.startswith("x_") and c[2:].isdigit()
        else c.replace("y_", "y", 1) if c.startswith("y_") and c[2:].isdigit()
        else c
        for c in df.columns
    ]

    df = df.apply(pd.to_numeric, errors="coerce")

    x_cols = [f"x{i}" for i in range(68)]
    y_cols = [f"y{i}" for i in range(68)]

    total_frames = len(df)

    df = df.dropna(subset=["confidence", "success"])

    valid = df[(df["success"] == 1) & (df["confidence"] >= 0.80)].copy()

    if len(valid) == 0:
        raise ValueError("No valid facial landmark frames detected.")

    valid = valid.dropna(subset=x_cols + y_cols)

    if len(valid) == 0:
        raise ValueError("No valid landmark coordinates found.")

    features = {}

    features["total_frames"] = total_frames
    features["valid_frames"] = len(valid)
    features["valid_frame_ratio"] = len(valid) / total_frames
    features["confidence_mean"] = df["confidence"].mean()

    x = valid[x_cols].values
    y = valid[y_cols].values

    eye_center_x = (
        x[:, 36:42].mean(axis=1) + x[:, 42:48].mean(axis=1)
    ) / 2

    eye_center_y = (
        y[:, 36:42].mean(axis=1) + y[:, 42:48].mean(axis=1)
    ) / 2

    eye_distance = np.sqrt(
        (x[:, 42:48].mean(axis=1) - x[:, 36:42].mean(axis=1)) ** 2
        + (y[:, 42:48].mean(axis=1) - y[:, 36:42].mean(axis=1)) ** 2
    )

    features["eye_distance_mean"] = np.mean(eye_distance)
    features["eye_distance_std"] = np.std(eye_distance)

    eye_distance[eye_distance == 0] = 1

    normalized_x = (x - eye_center_x[:, None]) / eye_distance[:, None]
    normalized_y = (y - eye_center_y[:, None]) / eye_distance[:, None]

    for i in range(68):
        features[f"landmark_x{i}_mean"] = np.mean(normalized_x[:, i])
        features[f"landmark_x{i}_std"] = np.std(normalized_x[:, i])
        features[f"landmark_y{i}_mean"] = np.mean(normalized_y[:, i])
        features[f"landmark_y{i}_std"] = np.std(normalized_y[:, i])

    features = {
        f"landmark_{key}": value
        for key, value in features.items()
    }

    return features
