from deepface import DeepFace

# WHY a function, not just a script?
# This file will be IMPORTED by classifier.py
# (just like extract.py was). We define a clean
# function here so other files can call
# detect_emotion(image_path) and get a result back.

def detect_emotion(image_path):
    """
    Detects the dominant emotion and full probability
    distribution for the face found in image_path.

    Returns a dict like:
    {
        'dominant_emotion': 'neutral',
        'emotion_scores': {
            'angry': 0.4, 'disgust': 0.0, 'fear': 1.2,
            'happy': 2.1, 'sad': 5.3, 'surprise': 0.1,
            'neutral': 90.9
        }
    }
    """

    # WHY DeepFace.analyze()?
    # This single function does THREE things internally:
    #   1. Finds the face in the image (face detection)
    #   2. Crops to just the face region
    #   3. Runs an emotion classifier (a CNN trained
    #      on millions of labelled faces) on that crop
    # actions=['emotion'] tells it to ONLY do emotion —
    # DeepFace can also do age, gender, race, but we
    # don't need those for this project.
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=True,   # raise error if no face found
            silent=True,              # suppress DeepFace's own print logs
        )
    except ValueError as e:
        # WHY catch ValueError specifically?
        # DeepFace raises ValueError when it can't
        # find ANY face in the image. Instead of
        # crashing the whole program, we return a
        # clean "no face" result so the rest of
        # the pipeline can continue gracefully —
        # e.g. for an image showing only the back
        # of someone's body.
        print(f" No face detected: {e}")
        return None
    
    # WHY result[0]?
    # DeepFace.analyze() ALWAYS returns a LIST,
    # because an image could contain multiple faces.
    # result[0] = the first detected face.
    # For now we assume one person per image —
    # we'll handle multiple people later when we
    # add multi-person support.
    face_result = result[0]

    # WHY round the scores?
    # face_result['emotion'] is a dict of 7 floats
    # that sum to 100 (it's a percentage distribution).
    # Raw values look like 0.00001234 or 87.345678 —
    # too many decimals to read. round(v, 2) keeps
    # 2 decimal places for clean display.
    emotion_scores = {
        k: round(float(v), 2)
        for k, v in face_result['emotion'].items()
    }

    return {
        'dominant_emotion': face_result['dominant_emotion'],
        'emotion_scores': emotion_scores,
    }


# WHY this test block?
# Same pattern as extract.py and classifier.py —
# lets us test THIS file alone before wiring it
# into the bigger pipeline. Run this file directly
# to check emotion detection works on my test image.

if __name__ == "__main__":
    IMAGE_PATH = "data/samples/test.jpg"

    print(f"Analysing emotion in: {IMAGE_PATH}")
    result = detect_emotion(IMAGE_PATH)

    if result is None:
        print("No face detected — nothing to show.")
    else:
        print(f"\nDominant emotion: {result['dominant_emotion']}")
        print("\nFull breakdown:")
        # sorted(..., reverse=True) shows highest scores first
        for emotion, score in sorted(
            result['emotion_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "*" * int(score / 5) + "+" * (20 - int(score / 5))
            print(f"{emotion:10s} {bar} {score:5.2f}%")