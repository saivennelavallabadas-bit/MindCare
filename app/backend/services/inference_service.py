
from pathlib import Path
import sys
import tempfile
import subprocess

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT))

from src.features.extract_audio import extract_audio_features
from src.features.extract_facial_aus import extract_facial_au_features
from src.features.extract_facial_landmarks import extract_landmark_features

MODEL_PATH = ROOT / "models" / "multimodal_missing_model.pt"
SCALER_PATH = ROOT / "models" / "multimodal_missing_scalers.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
FEATURE_SCHEMA = ROOT / "data" / "processed" / "multimodal_features.csv"

OPENFACE_PATH = Path(
    r"C:\Users\91988\Documents\FYP\Tools\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.audio_branch = nn.Sequential(
            nn.Linear(56, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.au_branch = nn.Sequential(
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.landmark_branch = nn.Sequential(
            nn.Linear(278, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.text_branch = nn.Sequential(
            nn.Linear(300, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(67, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, audio, au, landmark, text, mask):

        a = self.audio_branch(audio) * mask[:, 0:1]
        b = self.au_branch(au) * mask[:, 1:2]
        c = self.landmark_branch(landmark) * mask[:, 1:2]
        d = self.text_branch(text) * mask[:, 2:3]

        fused = torch.cat([a, b, c, d, mask], dim=1)

        return self.fusion(fused)


model = MultimodalNet().to(DEVICE)

state = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state)
model.eval()

scalers = joblib.load(SCALER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

schema = pd.read_csv(FEATURE_SCHEMA, nrows=1)

FEATURE_GROUPS = {
    name: [c for c in schema.columns if c.startswith(prefix)]
    for name, prefix in [
        ("audio", "audio_"),
        ("au", "au_"),
        ("landmark", "landmark_"),
        ("text", "text_")
    ]
}


def get_interpretation(score):

    if score < 5:
        return "Minimal symptom range"

    elif score < 10:
        return "Mild symptom range"

    elif score < 15:
        return "Moderate symptom range"

    elif score < 20:
        return "Moderately severe symptom range"

    return "Severe symptom range"


def extract_video_features(video_path, output_dir):

    if not OPENFACE_PATH.exists():
        raise FileNotFoundError(
            f"OpenFace executable not found: {OPENFACE_PATH}"
        )

    command = [
        str(OPENFACE_PATH),
        "-f", str(video_path),
        "-out_dir", str(output_dir)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(
            "OpenFace processing failed: "
            + result.stderr[-2000:]
        )

    csv_files = list(Path(output_dir).glob("*.csv"))

    if not csv_files:
        raise RuntimeError(
            "OpenFace did not generate a CSV feature file."
        )

    csv_path = csv_files[0]

    au_features = extract_facial_au_features(csv_path)
    landmark_features = extract_landmark_features(csv_path)

    return au_features, landmark_features


def prepare_scaled_features(name, features):

    columns = FEATURE_GROUPS[name]

    row = pd.DataFrame([features])

    if name == "landmark":

        row.columns = [
            "landmark_" + c if not c.startswith("landmark_") else c
            for c in row.columns
        ]

    else:

        row.columns = [
            name + "_" + c if not c.startswith(name + "_") else c
            for c in row.columns
        ]

    row = row.reindex(columns=columns)

    if row.isna().any().any():

        missing = row.columns[row.isna().any()].tolist()

        raise ValueError(
            f"Missing {name} features: {missing[:10]}"
        )

    values = scalers[name].transform(row.values)

    return values.astype(np.float32)


def empty_features(name):

    return np.zeros(
        (1, len(FEATURE_GROUPS[name])),
        dtype=np.float32
    )


def predict_screening(
    audio_path=None,
    video_path=None,
    transcript_text=None
):

    transcript_text = transcript_text or ""

    audio_available = audio_path is not None
    video_available = video_path is not None
    text_available = bool(transcript_text.strip())

    if not any([audio_available, video_available, text_available]):

        raise ValueError(
            "Upload at least one modality: audio, video or transcript."
        )

    audio_scaled = empty_features("audio")
    au_scaled = empty_features("au")
    landmark_scaled = empty_features("landmark")
    text_scaled = empty_features("text")

    if audio_available:

        audio_features = extract_audio_features(audio_path)

        audio_scaled = prepare_scaled_features(
            "audio", audio_features
        )

    if video_available:

        with tempfile.TemporaryDirectory() as temp_dir:

            au_features, landmark_features = extract_video_features(
                video_path,
                temp_dir
            )

        au_scaled = prepare_scaled_features(
            "au", au_features
        )

        landmark_scaled = prepare_scaled_features(
            "landmark", landmark_features
        )

    if text_available:

        text_vector = vectorizer.transform(
            [transcript_text]
        ).toarray().astype(np.float32)

        text_features = {
            "text_" + feature: value
            for feature, value in zip(
                vectorizer.get_feature_names_out(),
                text_vector[0]
            )
        }

        text_scaled = prepare_scaled_features(
            "text", text_features
        )

    mask = np.array(
        [[
            float(audio_available),
            float(video_available),
            float(text_available)
        ]],
        dtype=np.float32
    )

    inputs = {
        "audio": torch.tensor(
            audio_scaled,
            dtype=torch.float32,
            device=DEVICE
        ),
        "au": torch.tensor(
            au_scaled,
            dtype=torch.float32,
            device=DEVICE
        ),
        "landmark": torch.tensor(
            landmark_scaled,
            dtype=torch.float32,
            device=DEVICE
        ),
        "text": torch.tensor(
            text_scaled,
            dtype=torch.float32,
            device=DEVICE
        ),
        "mask": torch.tensor(
            mask,
            dtype=torch.float32,
            device=DEVICE
        )
    }

    with torch.no_grad():

        prediction = model(
            inputs["audio"],
            inputs["au"],
            inputs["landmark"],
            inputs["text"],
            inputs["mask"]
        ).cpu().numpy().flatten()[0]

    prediction = float(np.clip(prediction, 0, 24))

    modalities = []

    if audio_available:
        modalities.append("audio")

    if video_available:
        modalities.extend([
            "facial_action_units",
            "facial_landmarks"
        ])

    if text_available:
        modalities.append("text")

    return {
        "estimated_phq8_score": round(prediction, 2),
        "estimated_range": get_interpretation(prediction),
        "device": str(DEVICE),
        "model": "MindCare Missing-Modality Multimodal Regression Network",
        "modalities_used": modalities,
        "disclaimer": (
            "Research screening estimate only. "
            "Not a clinical diagnosis or validated clinical assessment."
        )
    }
