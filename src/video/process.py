# ─────────────────────────────────────────────
# WHY this file exists:
# Everything so far works on ONE image.
# This file takes a VIDEO, extracts individual
# frames at intervals, and runs your EXISTING
# pipeline (pose → emotion → Gemini) on each one.
# The result is a timeline of signals over time.
# ─────────────────────────────────────────────

import cv2
import time
import json
from pathlib import Path

from ultralytics import YOLO
from src.features.extract import extract_features
from src.detection.emotion import detect_emotion
from src.signals.classifier import analyse


def extract_sampled_frames(video_path, interval_seconds=2, output_dir="data/samples/frames"):
    """
    WHY this function?
    A video is just thousands of images played quickly.
    cv2.VideoCapture lets us read it frame by frame.
    Instead of processing EVERY frame (too slow, too
    many API calls), we save one frame every
    `interval_seconds` to disk as a JPG — exactly
    like the test.jpg you've been using.
    """

    # ─────────────────────────────────────────
    # WHY Path(output_dir).mkdir(...)?
    # Creates the output folder if it doesn't exist.
    # parents=True creates parent folders too.
    # exist_ok=True means "don't error if it
    # already exists" — safe to run multiple times.
    # ─────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # WHY cv2.VideoCapture(video_path)?
    # Opens the video file for reading, frame by
    # frame, similar to how open() opens a text file
    # for reading line by line.
    # ─────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)

    # ─────────────────────────────────────────
    # WHY cap.get(cv2.CAP_PROP_FPS)?
    # Every video has a "frames per second" rate —
    # e.g. 30fps means 30 images per second of footage.
    # We need this number to calculate WHICH frame
    # numbers correspond to "every 2 seconds".
    # ─────────────────────────────────────────
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ─────────────────────────────────────────
    # WHY this calculation?
    # If fps=30 and interval_seconds=2, then
    # frame_interval = 60. That means: save frame 0,
    # then frame 60, then frame 120, etc.
    # Frame 60 is exactly 2 seconds into the video
    # (60 frames / 30 fps = 2 seconds).
    # ─────────────────────────────────────────
    frame_interval = int(fps * interval_seconds)

    print(f"Video FPS: {fps:.1f}")
    print(f"Saving 1 frame every {interval_seconds}s (every {frame_interval} frames)")

    saved_frames = []   # will hold (frame_path, timestamp_seconds) pairs
    frame_count = 0

    # ─────────────────────────────────────────
    # WHY this while loop?
    # cap.read() returns TWO things:
    #   - ret: True if a frame was read, False if
    #     the video ended (no more frames)
    #   - frame: the actual image data (numpy array)
    # We loop until ret is False (video finished).
    # ─────────────────────────────────────────
    while True:
        ret, frame = cap.read()

        if not ret:
            break   # video ended

        # ─────────────────────────────────────
        # WHY frame_count % frame_interval == 0?
        # % is the modulo operator — it gives the
        # REMAINDER after division. If frame_interval=60,
        # then frame_count % 60 == 0 is True for
        # frame_count = 0, 60, 120, 180, ...
        # This is how we pick "every 60th frame"
        # without saving everything.
        # ─────────────────────────────────────
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps   # seconds into the video
            frame_path = f"{output_dir}/frame_{frame_count:05d}.jpg"

            # ─────────────────────────────────
            # WHY cv2.imwrite()?
            # Saves the in-memory frame (numpy array)
            # to disk as a JPG file. Your existing
            # pipeline (YOLO, DeepFace, Gemini) all
            # expect a FILE PATH, not raw frame data —
            # so we save each sampled frame to disk
            # to reuse that exact same code.
            # ─────────────────────────────────
            cv2.imwrite(frame_path, frame)
            saved_frames.append((frame_path, timestamp))
            print(f"  Saved frame at {timestamp:.1f}s → {frame_path}")

        frame_count += 1

    cap.release()   # WHY: releases the video file handle, frees memory

    print(f"\n✅ Extracted {len(saved_frames)} frames from video")
    return saved_frames


def process_video(video_path, interval_seconds=2):
    """
    WHY this function?
    This is the main orchestrator. It:
      1. Extracts sampled frames from the video
      2. Runs your FULL existing pipeline on each frame
      3. Adds a small delay between API calls (rate limit)
      4. Collects everything into a timeline
    """

    print(f"Processing video: {video_path}\n")
    print("=" * 50)

    # Step 1 — get sampled frames
    frames = extract_sampled_frames(video_path, interval_seconds)

    # ─────────────────────────────────────────
    # WHY load YOLO once, outside the loop?
    # Loading a model takes time (reading the .pt
    # file, setting up the network). If we loaded
    # it INSIDE the loop, we'd reload it for every
    # single frame — wasting seconds per frame.
    # Load once, reuse for all frames.
    # ─────────────────────────────────────────
    yolo = YOLO("yolov8n-pose.pt")

    timeline = []   # will hold one result dict per frame

    print("\n" + "=" * 50)
    print("Analysing frames...")
    print("=" * 50)

    for i, (frame_path, timestamp) in enumerate(frames):
        print(f"\n[{i+1}/{len(frames)}] Frame at {timestamp:.1f}s")

        # ─────────────────────────────────────
        # WHY verbose=False?
        # YOLO normally prints a line for every
        # inference. With 30 frames that's 30 lines
        # of noise. verbose=False silences it so
        # our own print statements stay readable.
        # ─────────────────────────────────────
        results = yolo(frame_path, conf=0.5, verbose=False)

        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            print("  ⚠️  No person detected — skipping")
            # ─────────────────────────────────
            # WHY append a "skipped" entry instead
            # of just `continue`?
            # If we silently skip, the timeline has
            # GAPS — confusing when plotting later.
            # Recording "no_person: True" preserves
            # the timestamp slot so a chart can show
            # "no data" instead of missing entirely.
            # ─────────────────────────────────
            timeline.append({
                "timestamp": round(timestamp, 1),
                "no_person": True,
            })
            continue

        # Step 2a — pose features (same as before)
        kps = results[0].keypoints.xy[0].cpu().numpy()
        features = extract_features(kps)

        # Step 2b — emotion (same as before)
        emotion = detect_emotion(frame_path)

        # Step 2c — Gemini analysis (same as before)
        result = analyse(frame_path, features, emotion)

        # ─────────────────────────────────────
        # WHY add timestamp to the result dict?
        # result already has confidence/stress/etc.
        # but doesn't know WHEN in the video this
        # was. Adding timestamp lets us build a
        # time-series chart later (x-axis = time,
        # y-axis = confidence score).
        # ─────────────────────────────────────
        result["timestamp"] = round(timestamp, 1)
        result["no_person"] = False

        timeline.append(result)

        print(f"  Confidence: {result.get('confidence', 'N/A')}  "
              f"Stress: {result.get('stress', 'N/A')}  "
              f"Engagement: {result.get('engagement', 'N/A')}")

        # ─────────────────────────────────────
        # WHY time.sleep(4)?
        # Gemini free tier allows 15 requests/minute
        # = 1 request every 4 seconds MAXIMUM.
        # Sleeping 4 seconds between calls keeps us
        # safely under that limit, even with DeepFace
        # and YOLO processing time added on top.
        # ─────────────────────────────────────
        if i < len(frames) - 1:   # don't sleep after the last frame
            time.sleep(4)

    return timeline


def save_timeline(timeline, output_path="data/samples/timeline.json"):
    """
    WHY save to JSON?
    The timeline is a list of dicts — exactly the
    kind of data JSON is designed for. Saving it
    means you can load it later for the web UI,
    or plot it, WITHOUT re-running the entire
    video analysis (which takes minutes).
    """
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\n✅ Timeline saved to: {output_path}")


def print_timeline_summary(timeline):
    """
    WHY a summary view?
    30 full Gemini reports is a LOT of text.
    This prints just the key numbers per frame —
    a quick scan of how signals change over time.
    """
    print("\n" + "=" * 60)
    print("  TIMELINE SUMMARY")
    print("=" * 60)
    print(f"  {'Time':>6s}  {'Confidence':>10s}  {'Stress':>8s}  {'Engagement':>10s}  Dominant signal")
    print("-" * 60)

    for entry in timeline:
        if entry.get("no_person"):
            print(f"  {entry['timestamp']:>5.1f}s  {'--':>10s}  {'--':>8s}  {'--':>10s}  (no person detected)")
        else:
            print(f"  {entry['timestamp']:>5.1f}s  "
                  f"{entry.get('confidence', 0):>10.2f}  "
                  f"{entry.get('stress', 0):>8.2f}  "
                  f"{entry.get('engagement', 0):>10.2f}  "
                  f"{entry.get('dominant_signal', 'N/A')}")

    print("=" * 60)


# ─────────────────────────────────────────────
# WHY if __name__ == '__main__'?
# Same pattern as before — lets us test this
# file directly: python -m src.video.process
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = "data/samples/test_video.mp4"

    if not Path(VIDEO_PATH).exists():
        print(f"❌ No video found at {VIDEO_PATH}")
        print("   Add a short video (10-30 seconds works well for testing)")
    else:
        timeline = process_video(VIDEO_PATH, interval_seconds=2)
        save_timeline(timeline)
        print_timeline_summary(timeline)