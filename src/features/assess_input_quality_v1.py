
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")

OUTPUT = Path("results/input_quality_v1")
OUTPUT.mkdir(parents=True, exist_ok=True)


def assess_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)

    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]

    low_energy_ratio = float(np.mean(rms < 0.01))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.99))
    rms_mean = float(np.mean(rms))
    rms_db = float(20 * np.log10(rms_mean + 1e-10))

    warnings_list = []

    if low_energy_ratio > 0.50:
        warnings_list.append("High proportion of low-energy frames")

    if duration < 10:
        warnings_list.append("Audio recording is very short")

    if clipping_ratio > 0.01:
        warnings_list.append("Audio clipping detected")

    if rms_db < -40:
        warnings_list.append("Audio signal is very quiet")

    return {
        "available": True,
        "duration_seconds": round(duration, 2),
        "sample_rate": int(sr),
        "low_energy_ratio": round(low_energy_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "rms_mean": round(rms_mean, 6),
        "rms_db": round(rms_db, 2),
        "warnings": warnings_list
    }


def assess_facial(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()

    required = ["timestamp", "confidence", "success"]

    for col in required:
        if col not in d.columns:
            raise ValueError(f"Missing required OpenFace column: {col}")

    d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce")
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
    d["success"] = pd.to_numeric(d["success"], errors="coerce")

    d = d.dropna(subset=required)

    if len(d) == 0:
        raise ValueError("No valid facial tracking rows found")

    total_frames = len(d)
    success_rate = float(d["success"].mean())
    mean_confidence = float(d["confidence"].mean())
    low_confidence_frames = int((d["confidence"] < 0.80).sum())
    failed_frames = int((d["success"] == 0).sum())
    duration = float(d["timestamp"].max())

    warnings_list = []

    if success_rate < 0.90:
        warnings_list.append("Low facial tracking success rate")

    if mean_confidence < 0.80:
        warnings_list.append("Low average facial tracking confidence")

    if duration < 60:
        warnings_list.append("Facial recording is very short")

    if low_confidence_frames > 0:
        warnings_list.append("Low-confidence facial tracking frames detected")

    return {
        "available": True,
        "source": "OpenFace Action Unit file",
        "total_frames": int(total_frames),
        "duration_seconds": round(duration, 2),
        "tracking_success_rate": round(success_rate, 4),
        "mean_confidence": round(mean_confidence, 4),
        "low_confidence_frames": low_confidence_frames,
        "failed_frames": failed_frames,
        "warnings": warnings_list
    }


def assess_text(path):
    d = pd.read_csv(path, sep="\t")
    d.columns = d.columns.str.strip()

    if "speaker" not in d.columns or "value" not in d.columns:
        raise ValueError("Transcript must contain speaker and value columns")

    participant = d[
        d["speaker"].astype(str).str.strip().str.lower() == "participant"
    ]

    text = " ".join(
        participant["value"].fillna("").astype(str).tolist()
    ).strip()

    all_text = " ".join(
        d["value"].fillna("").astype(str).tolist()
    ).strip()

    word_count = len(re.findall(r"\b\w+\b", text))
    sentence_count = len(re.findall(r"[.!?]+", text))
    participant_turns = len(participant)

    warnings_list = []

    if word_count < 50:
        warnings_list.append("Very short participant transcript")

    if participant_turns == 0:
        warnings_list.append("No participant speech found")

    return {
        "available": True,
        "characters": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "participant_turns": participant_turns,
        "total_transcript_turns": int(len(d)),
        "all_speaker_characters": len(all_text),
        "warnings": warnings_list
    }


def calculate_quality_score(report):
    scores = {}

    if report["audio"]["available"]:
        a = report["audio"]
        score = 0

        if a["duration_seconds"] >= 60:
            score += 25

        if a["sample_rate"] >= 16000:
            score += 25

        if a["low_energy_ratio"] <= 0.50:
            score += 25

        if a["rms_db"] >= -40:
            score += 25

        scores["audio"] = score

    if report["facial"]["available"]:
        f = report["facial"]
        score = 0

        if f["duration_seconds"] >= 60:
            score += 25

        if f["total_frames"] >= 1000:
            score += 25

        if f["tracking_success_rate"] >= 0.90:
            score += 25

        if f["mean_confidence"] >= 0.80:
            score += 25

        scores["facial"] = score

    if report["text"]["available"]:
        t = report["text"]
        score = 0

        if t["word_count"] >= 500:
            score += 50

        if t["participant_turns"] >= 20:
            score += 50

        scores["text"] = score

    return scores


def main():
    print("=" * 65)
    print("MINDCARE - INPUT QUALITY ASSESSMENT V1")
    print("=" * 65)

    audio_path = input("Audio file path (Enter to skip): ").strip()
    facial_path = input("OpenFace AU file path (Enter to skip): ").strip()
    transcript_path = input("Transcript file path (Enter to skip): ").strip()

    report = {
        "audio": {"available": False},
        "facial": {"available": False},
        "text": {"available": False},
        "overall_warnings": [],
        "modalities_available": [],
        "quality_scores": {},
        "overall_quality_score": None,
        "overall_status": "No inputs provided",
        "score_note": (
            "Technical usability score based on prototype rules. "
            "Not a clinical measure or model confidence score."
        )
    }

    if audio_path:
        try:
            report["audio"] = assess_audio(audio_path)
        except Exception as e:
            report["audio"] = {
                "available": False,
                "error": str(e)
            }

    if facial_path:
        try:
            report["facial"] = assess_facial(facial_path)
        except Exception as e:
            report["facial"] = {
                "available": False,
                "error": str(e)
            }

    if transcript_path:
        try:
            report["text"] = assess_text(transcript_path)
        except Exception as e:
            report["text"] = {
                "available": False,
                "error": str(e)
            }

    available = []

    for modality in ["audio", "facial", "text"]:
        if report[modality].get("available", False):
            available.append(modality)

            for warning in report[modality].get("warnings", []):
                report["overall_warnings"].append(
                    f"{modality}: {warning}"
                )

    report["modalities_available"] = available

    scores = calculate_quality_score(report)
    report["quality_scores"] = scores

    if scores:
        overall = round(sum(scores.values()) / len(scores))
        report["overall_quality_score"] = overall

        if overall >= 80:
            report["overall_status"] = "Good technical quality"
        elif overall >= 60:
            report["overall_status"] = "Review recommended"
        else:
            report["overall_status"] = "Poor technical quality"

    output_file = OUTPUT / "quality_report.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("QUALITY ASSESSMENT REPORT")
    print("=" * 65)

    print(json.dumps(report, indent=4, ensure_ascii=False))

    print(f"\nReport saved: {output_file}")


if __name__ == "__main__":
    main()
