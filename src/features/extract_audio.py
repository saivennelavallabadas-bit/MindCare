from pathlib import Path
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

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


if __name__ == "__main__":
    audio_path = Path(
        r"C:\Users\91988\Documents\FYP\DAIC-WOC dataset\300_P\300_AUDIO.wav"
    )

    result = extract_audio_features(audio_path)

    print("=" * 60)
    print("SPEECH FEATURE EXTRACTION TEST")
    print("=" * 60)
    print("Audio:", audio_path.name)
    print("Features extracted:", len(result))

    for key, value in result.items():
        print(f"{key}: {value:.6f}")

    print("\nTEST COMPLETED")