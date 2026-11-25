from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routers.quality import router as quality_router

app = FastAPI(
    title="MindCare API",
    description="Experimental Multimodal Depression Screening API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quality_router)


@app.get("/")
def home():
    return {
        "application": "MindCare",
        "message": "MindCare API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "MindCare"
    }