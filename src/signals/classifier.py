# ─────────────────────────────────────────────
# WHY this file exists:
# extract.py gives us numbers like spine_tilt=4.93
# Those numbers alone mean nothing to a non-expert.
# This file sends those numbers + the actual image
# to Gemini and gets back a human-readable analysis.
# Gemini acts as the "brain" that interprets the math.
# ─────────────────────────────────────────────

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from src.detection.emotion import detect_emotion

# ─────────────────────────────────────────────
# WHY load_dotenv()?
# Reads your .env file and loads GEMINI_API_KEY
# into os.environ so we can safely access it.
# Key never hardcoded = safe to push to GitHub.
# ─────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Check your .env file has: GEMINI_API_KEY=AIza..."
    )

# ─────────────────────────────────────────────
# WHY genai.Client()?
# The new SDK uses a unified Client object to handle
# all API operations. Passing api_key explicitly
# registers it for every call made through `client`.
# ─────────────────────────────────────────────
client = genai.Client(api_key=api_key)


def normalise_scores(result):
    # ─────────────────────────────────────────
    # WHY this function exists:
    # Even with a response_schema forcing the TYPE
    # to be a number, Gemini can still return that
    # number on the wrong SCALE — e.g. 45 instead
    # of 0.45, if it "thinks" in percentages for an
    # ambiguous frame. The schema guarantees
    # "this is a number", not "this number is
    # between 0 and 1".
    #
    # Rather than hoping the prompt is followed
    # perfectly every time, we FIX the data in code.
    # This is a defensive programming pattern:
    # never fully trust external input, even from
    # an AI model with a schema.
    # ─────────────────────────────────────────
    for key in ["confidence", "stress", "engagement"]:
        if key in result:
            value = result[key]

            # ─────────────────────────────────
            # WHY check value > 1?
            # If Gemini returned 45 (meant as 45%),
            # dividing by 100 gives 0.45 — correct.
            # If Gemini returned 0.45 (already correct),
            # this check is False, value stays 0.45.
            # This single line fixes BOTH cases.
            # ─────────────────────────────────
            if value > 1:
                value = value / 100

            # ─────────────────────────────────
            # WHY clamp with min/max?
            # Extra safety: even after dividing by
            # 100, if something gives 150 → 1.5,
            # this forces it back to 1.0. Values
            # below 0 (shouldn't happen, but just
            # in case) get forced to 0.0.
            # min(1.0, max(0.0, value)) reads as:
            # "take value, but never below 0,
            #  and never above 1"
            # ─────────────────────────────────
            result[key] = round(min(1.0, max(0.0, value)), 2)

    return result


def analyse(image_path, features, emotion=None):
    # ─────────────────────────────────────────
    # WHY PIL.Image.open()?
    # Gemini's Python library accepts PIL Image
    # objects directly — no base64 conversion needed.
    # PIL opens the image file into memory seamlessly.
    # ─────────────────────────────────────────
    print("Loading image...")
    image = Image.open(image_path)

    # ─────────────────────────────────────────
    # WHY json.dumps(features, indent=2)?
    # Converts our Python dict of features into
    # a clean, readable JSON string to paste into
    # the prompt so Gemini parses each key-value pair.
    # ─────────────────────────────────────────
    features_text = json.dumps(features, indent=2)

    # ─────────────────────────────────────────
    # WHY build emotion_text conditionally?
    # If a face WAS detected, we format the
    # emotion data as JSON, same as features.
    # If NOT (emotion is None), we tell Gemini
    # explicitly that no face data is available —
    # so it doesn't assume or hallucinate an emotion.
    # ─────────────────────────────────────────
    if emotion:
        emotion_text = json.dumps(emotion, indent=2)
    else:
        emotion_text = "No face detected — emotion data unavailable."

    # ─────────────────────────────────────────
    # WHY still describe the JSON shape in the prompt
    # even though response_schema enforces it?
    # response_schema guarantees the STRUCTURE
    # (field names, types). It does NOT explain
    # what each field MEANS or what RANGE the
    # numbers should be in. The prompt still needs
    # to tell Gemini "confidence is 0.0-1.0", even
    # though normalise_scores() is our backup if
    # Gemini ignores that instruction.
    # ─────────────────────────────────────────
    prompt = f"""You are an expert body language analyst with deep knowledge
of nonverbal communication research.

You will be given an image of a person and precise postural measurements
extracted from their pose keypoints.

Your job is to interpret what these signals mean TOGETHER — not in isolation.
A single feature means little; the combination tells the real story.

POSTURAL MEASUREMENTS:
{features_text}

FACIAL EMOTION ANALYSIS:
{emotion_text}

KEY TO READING THESE NUMBERS:
- spine_tilt      : degrees from vertical. Under 10° = upright. Over 20° = slouching.
- forward_lean    : normalised ratio. Positive = leaning toward camera. Negative = leaning back.
- shoulder_symmetry: 0 = perfectly level. High value = uneven shoulders (tension signal).
- shoulder_elevation: high value = shoulders raised toward ears (stress signal).
- arm_openness    : positive = arms open/spread. Negative = arms crossed or tucked in.
- avg_wrist_height: positive = hands raised/active. Negative = hands resting low.
- head_tilt       : 0 = head centred. Non-zero = head tilting left or right.

Analyse the image alongside these measurements AND the facial emotion data
(if available).

IMPORTANT: confidence, stress, and engagement must each be a float between
0.0 and 1.0 (e.g. 0.75), NEVER a percentage like 75."""

    print("Sending to Gemini API...")

    # ─────────────────────────────────────────
    # WHY config=types.GenerateContentConfig(...)?
    # Instead of asking for JSON text and stripping
    # markdown fences ourselves, the new SDK lets us
    # pass a response_schema. This forces Gemini to
    # return a clean JSON object matching this exact
    # structure — no markdown, no preamble text.
    # ─────────────────────────────────────────
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "confidence": types.Schema(type=types.Type.NUMBER),
                        "stress": types.Schema(type=types.Type.NUMBER),
                        "engagement": types.Schema(type=types.Type.NUMBER),
                        "dominant_signal": types.Schema(type=types.Type.STRING),
                        "key_observations": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING)
                        ),
                        "recommendation": types.Schema(type=types.Type.STRING),
                    },
                    required=["confidence", "stress", "engagement",
                              "dominant_signal", "key_observations", "recommendation"]
                ),
            ),
        )

        # ─────────────────────────────────────
        # WHY response.text (not raw_text)?
        # response.text is the attribute the SDK
        # actually returns — your previous version
        # referenced `raw_text`, which was never
        # defined. With response_schema set, this
        # text is GUARANTEED to be valid JSON —
        # no markdown fences, no extra preamble.
        # ─────────────────────────────────────
        result = json.loads(response.text)

        # ─────────────────────────────────────
        # WHY still call normalise_scores()?
        # response_schema guarantees confidence is
        # a NUMBER. It does NOT guarantee that number
        # is between 0 and 1. normalise_scores()
        # is our scale safety net — see its docstring
        # above for the full reasoning.
        # ─────────────────────────────────────
        result = normalise_scores(result)
        return result

    except Exception as e:
        # ─────────────────────────────────────
        # WHY catch Exception broadly here?
        # Many things can go wrong: network errors,
        # API quota errors, malformed JSON, missing
        # fields. Rather than crashing the whole
        # video pipeline on ONE bad frame, we return
        # a safe fallback dict with the SAME KEYS
        # the rest of the code expects. This means
        # process_video() can keep going to the
        # next frame instead of stopping entirely.
        # ─────────────────────────────────────
        print(f"An unexpected error occurred during API call or parsing: {e}")
        return {
            "raw": str(e),
            "confidence": 0.0, "stress": 0.0, "engagement": 0.0,
            "dominant_signal": "Error processing data",
            "key_observations": ["Could not parse response"],
            "recommendation": "Check API logs.",
        }


def print_report(result):
    # ─────────────────────────────────────────
    # WHY check len(result) == 1 alongside "raw"?
    # Our fallback dict above includes "raw" PLUS
    # all the normal keys (confidence, stress, etc.)
    # so print_report can still display something
    # useful even on error. len(result) == 1 would
    # only be true for a truly bare {"raw": ...}
    # dict — which no longer happens, but this guards
    # against any future fallback that's minimal.
    # ─────────────────────────────────────────
    if "raw" in result and len(result) == 1:
        print(result["raw"])
        return

    def bar(score):
        filled = int(score * 10)
        return "█" * filled + "░" * (10 - filled)

    print("\n" + "=" * 52)
    print("   TELLTRACE — BODY LANGUAGE ANALYSIS")
    print("=" * 52)
    print(f"\n  Dominant signal : {result.get('dominant_signal', 'N/A')}")
    print(f"\n  Confidence  {bar(result['confidence'])}  {result['confidence']:.2f}")
    print(f"  Stress      {bar(result['stress'])}  {result['stress']:.2f}")
    print(f"  Engagement  {bar(result['engagement'])}  {result['engagement']:.2f}")
    print("\n  Key observations:")
    for obs in result.get("key_observations", []):
        print(f"    • {obs}")
    print(f"\n  Recommendation:")
    print(f"    {result.get('recommendation', 'N/A')}")
    print("=" * 52)


if __name__ == "__main__":
    from src.features.extract import extract_features
    from ultralytics import YOLO

    IMAGE_PATH = "data/samples/test.jpg"

    # Step 1 — detect and extract features
    yolo    = YOLO("yolov8n-pose.pt")
    results = yolo(IMAGE_PATH, conf=0.5, verbose=False)

    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        print("No person detected in image.")
    else:
        kps      = results[0].keypoints.xy[0].cpu().numpy()
        features = extract_features(kps)

        print("Posture features extracted:")
        for k, v in features.items():
            print(f"  {k:25s}: {v}")
        print()

        # ─────────────────────────────────────
        # WHY call detect_emotion here?
        # This runs the DeepFace analysis on the
        # SAME image, in the SAME pipeline run.
        # If it returns None (no face), emotion
        # stays None and analyse() handles it
        # gracefully — no crash.
        # ─────────────────────────────────────
        print("Detecting facial emotion...")
        emotion = detect_emotion(IMAGE_PATH)

        if emotion:
            print(f"Dominant emotion: {emotion['dominant_emotion']}")
        else:
            print("No face detected.")
        print()

        # Step 2 — send to Gemini (now with emotion too)
        result = analyse(IMAGE_PATH, features, emotion)

        # Step 3 — display report
        print_report(result)