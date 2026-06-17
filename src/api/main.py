# WHY this file exists:
# Everything I've built so far runs from the
# command line. This file exposes the SAME
# pipeline as HTTP endpoints — so a web browser
# (or any frontend) can upload an image/video
# and get results back as JSON.


import os
import shutil
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from src.features.extract import extract_features
from src.detection.emotion import detect_emotion
from src.signals.classifier import analyse
from src.video.process import process_video

# WHY FastAPI()?
# This creates the "app" object — the central
# object FastAPI uses to register all my
# endpoints (routes) and middleware.
app = FastAPI(title="TellTrace API")

# WHY CORSMiddleware?
# CORS = Cross-Origin Resource Sharing.
# By default, browsers BLOCK a webpage from
# calling an API on a different port/domain
# (a security feature). My frontend (e.g.
# localhost:5500) and backend (localhost:8000)
# are technically "different origins".
# This middleware tells the browser: "it's OK,
# allow requests from anywhere" — fine for local
# development, I'd restrict this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WHY load YOLO once, at startup?
# Same reason as in process.py — loading the
# model takes time. If we loaded it INSIDE each
# endpoint function, every API call would be
# slow. Loading it once when the server starts
# means every request reuses the same loaded
# model — fast.
yolo_model = YOLO("yolov8n-pose.pt")

# WHY these folders?
# Uploaded files need to be saved to disk
# temporarily (My pipeline functions expect
# file PATHS, not raw bytes). UPLOAD_DIR is
# where we save them before processing.
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
    # WHY a root endpoint?
    # A simple "is the server alive?" check.
    # Visiting http://localhost:8000 in a browser
    # should show this message — confirms the
    # server is running before I build anything else.
    return {"status": "TellTrace API is running"}


@app.post("/analyse-image")
async def analyse_image(file: UploadFile = File(...)):
    """
    WHY async def?
    FastAPI endpoints that handle file uploads
    should be 'async' — this lets the server
    handle OTHER requests while waiting for the
    file to finish uploading, instead of blocking
    everything. I don't need to fully understand
    async yet — just know it's the standard pattern
    for file upload endpoints in FastAPI.

    WHY UploadFile = File(...)?
    This tells FastAPI: "expect a file in the
    request named 'file'". The ... means it's
    REQUIRED — the request fails with a clear
    error if no file is sent.
    """
    # WHY save to disk first?
    # file.file is the raw uploaded data in memory.
    # My pipeline functions (extract_features,
    # detect_emotion, analyse) all expect a FILE
    # PATH on disk — so we write the upload to
    # disk first, then pass that path to my
    # EXISTING, UNCHANGED functions.
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Received image: {save_path}")
    # WHY this exact sequence?
    # This is IDENTICAL to the test block at the
    # bottom of classifier.py — same 3 steps:
    # detect pose → extract features → detect
    # emotion → analyse with Gemini.
    # We're just calling my EXISTING functions,
    # not rewriting any logic.

    results = yolo_model(str(save_path), conf=0.5, verbose=False)

    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return JSONResponse(
            status_code=200,
            content={"error": "No person detected in image"}
        )

    kps = results[0].keypoints.xy[0].cpu().numpy()
    features = extract_features(kps)

    emotion = detect_emotion(str(save_path))

    result = analyse(str(save_path), features, emotion)
    # WHY add features and emotion to the response?
    # The frontend might want to display the raw
    # posture numbers and emotion breakdown too,
    # not just Gemini's summary. Bundling
    # everything into one response means the
    # frontend doesn't need a second API call.
    return {
        "features": features,
        "emotion": emotion,
        "analysis": result,
    }


@app.post("/analyse-video")
async def analyse_video_endpoint(file: UploadFile = File(...)):
    """
    WHY a separate endpoint for video?
    Video processing takes much longer (remember
    the 4-second sleep between frames for Gemini's
    rate limit). Separating image and video endpoints
    means the frontend can show different loading
    states — "Analysing..." vs "Analysing video,
    this may take a minute...".
    """

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Received video: {save_path}")

    # WHY call process_video() directly?
    # This is my EXISTING function from
    # src/video/process.py — completely
    # unchanged. It already does everything:
    # sampling, pose, emotion, Gemini, timeline.
    timeline = process_video(str(save_path), interval_seconds=2)

    return {"timeline": timeline}


# WHY mount static files?
# This serves my frontend HTML/CSS/JS files
# directly from FastAPI — so I don't need a
# SEPARATE web server for the frontend. Visiting
# http://localhost:8000/app/ will serve
# src/api/static/index.html
static_dir = Path("src/api/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")