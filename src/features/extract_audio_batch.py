from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/participant_manifest.csv"
OUTPUT = ROOT / "data/processed/audio_features.csv"
ERRORS = ROOT / "reports/tables/audio_extraction_errors.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ERRORS.parent.mkdir(parents=True, exist_ok=True)


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    features = {}

    features["duration"] = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr)[0]

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    features["centroid_mean"] = np.mean(centroid)
    features["centroid_std"] = np.std(centroid)

    features["bandwidth_mean"] = np.mean(bandwidth)
    features["bandwidth_std"] = np.std(bandwidth)

    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_std"] = np.std(rolloff)

    features["contrast_mean"] = np.mean(contrast)
    features["contrast_std"] = np.std(contrast)

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    features["tempo"] = np.mean(tempo)

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    return features


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["audio_available"] == True].copy()

    results = []
    errors = []

    total = len(df)

    print("=" * 65)
    print("MULTIMODAL FYP - BATCH AUDIO FEATURE EXTRACTION")
    print("=" * 65)
    print("Participants:", total)

    for index, row in df.iterrows():
        participant_id = row["participant_id"]
        audio_path = row["audio_path"]

        try:
            features = extract_audio_features(audio_path)

            results.append({
                "participant_id": participant_id,
                "split": row["split"],
                **features
            })

        except Exception as e:
            errors.append({
                "participant_id": participant_id,
                "audio_path": audio_path,
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
    print("EXTRACTION FINISHED")
    print("=" * 65)
    print("Successful:", len(results))
    print("Failed:", len(errors))
    print("Feature table shape:", feature_df.shape)
    print("Features saved:", OUTPUT)
    print("Errors saved:", ERRORS)


if __name__ == "__main__":
    main()