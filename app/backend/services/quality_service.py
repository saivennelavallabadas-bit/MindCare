from src.features.assess_input_quality_v1 import (
    assess_audio,
    assess_facial,
    assess_text,
    calculate_quality_score
)


def assess_inputs(audio_path=None, facial_path=None, transcript_path=None):

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

    return report