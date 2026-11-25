import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from fastapi import APIRouter, UploadFile, HTTPException, Form, File

from app.backend.services.quality_service import assess_inputs
from app.backend.services.inference_service import predict_screening


router = APIRouter(prefix="/quality", tags=["Input Quality and Screening"])


@router.post("/assess")
async def assess_uploaded_files(
    audio: Optional[UploadFile] = File(None),
    facial: Optional[UploadFile] = File(None),
    transcript: Optional[UploadFile] = File(None)
):
    files = {
        "audio": audio,
        "facial": facial,
        "transcript": transcript
    }

    if not any(files.values()):
        raise HTTPException(status_code=400, detail="Upload at least one file")

    temp_paths = {}

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            for name, uploaded_file in files.items():

                if uploaded_file is not None:

                    file_path = Path(temp_dir) / uploaded_file.filename
                    content = await uploaded_file.read()

                    if not content:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{name} file is empty"
                        )

                    file_path.write_bytes(content)
                    temp_paths[name] = file_path

            report = assess_inputs(
                audio_path=temp_paths.get("audio"),
                facial_path=temp_paths.get("facial"),
                transcript_path=temp_paths.get("transcript")
            )

            return report

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def read_transcript_file(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".txt":

        return file_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

    if suffix == ".csv":

        df = pd.read_csv(file_path, sep=None, engine="python")

        df.columns = df.columns.str.strip().str.lower()

        if "speaker" in df.columns and "value" in df.columns:

            participant = df[
                df["speaker"].astype(str).str.strip().str.lower()
                == "participant"
            ]

            return " ".join(
                participant["value"].dropna().astype(str)
            ).strip()

        if "value" in df.columns:

            return " ".join(
                df["value"].dropna().astype(str)
            ).strip()

        return " ".join(
            df.astype(str).fillna("").values.flatten()
        ).strip()

    raise ValueError("Transcript must be a .txt or .csv file")


@router.post("/predict")
async def predict_uploaded_files(
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    transcript: Optional[UploadFile] = File(None),
    patient_id: str = Form("Not provided")
):

    if not any([audio, video, transcript]):

        raise HTTPException(
            status_code=400,
            detail="Upload at least one modality"
        )

    if audio and (
        not audio.filename
        or not audio.filename.lower().endswith(".wav")
    ):

        raise HTTPException(
            status_code=400,
            detail="Audio must be a WAV file"
        )

    if video and (
        not video.filename
        or not video.filename.lower().endswith(
            (".mp4", ".avi", ".mov")
        )
    ):

        raise HTTPException(
            status_code=400,
            detail="Video must be an MP4, AVI or MOV file"
        )

    if transcript and (
        not transcript.filename
        or not transcript.filename.lower().endswith(
            (".txt", ".csv")
        )
    ):

        raise HTTPException(
            status_code=400,
            detail="Transcript must be a TXT or CSV file"
        )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            audio_path = None
            video_path = None
            transcript_text = ""

            if audio:

                audio_content = await audio.read()

                if not audio_content:

                    raise HTTPException(
                        status_code=400,
                        detail="Audio file is empty"
                    )

                audio_path = temp_dir / "uploaded_audio.wav"
                audio_path.write_bytes(audio_content)

            if video:

                video_content = await video.read()

                if not video_content:

                    raise HTTPException(
                        status_code=400,
                        detail="Video file is empty"
                    )

                video_path = temp_dir / (
                    "uploaded_video" + Path(video.filename).suffix
                )

                video_path.write_bytes(video_content)

            if transcript:

                transcript_content = await transcript.read()

                if not transcript_content:

                    raise HTTPException(
                        status_code=400,
                        detail="Transcript file is empty"
                    )

                transcript_path = temp_dir / (
                    "uploaded_transcript" + Path(transcript.filename).suffix
                )

                transcript_path.write_bytes(transcript_content)

                transcript_text = read_transcript_file(
                    transcript_path
                )

                if not transcript_text.strip():

                    raise HTTPException(
                        status_code=400,
                        detail="Transcript contains no usable text"
                    )

            prediction = predict_screening(
                audio_path=str(audio_path) if audio_path else None,
                video_path=str(video_path) if video_path else None,
                transcript_text=transcript_text
            )

            uploaded_modalities = []

            if audio:
                uploaded_modalities.append("audio")

            if video:
                uploaded_modalities.append("video")

            if transcript:
                uploaded_modalities.append("transcript")

            return {
                "patient_id": patient_id,
                "uploaded_modalities": uploaded_modalities,
                "estimated_phq8_score": prediction["estimated_phq8_score"],
                "estimated_range": prediction["estimated_range"],
                "model": prediction["model"],
                "device": prediction["device"],
                "disclaimer": prediction["disclaimer"]
            }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))