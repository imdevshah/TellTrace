import numpy as np
from ultralytics import YOLO

LABELS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def run_detection(model_path, image_path):
    """
    WHY a single reusable function?
    Both models go through IDENTICAL steps — load,
    detect, extract keypoints and confidences. Writing
    this once and calling it twice (once per model)
    guarantees a fair comparison — no accidental
    differences in how each model is run.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=0.5, verbose=False)

    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    # WHY .xy AND .conf?
    # .xy gives the (x, y) pixel position of each
    # of the 17 keypoints — WHERE the model thinks
    # each joint is.
    # .conf gives a confidence score (0-1) for EACH
    # keypoint — HOW SURE the model is about that
    # position. A fine-tuned model should generally
    # show higher confidence on clearly visible joints.

    keypoints = results[0].keypoints.xy[0].cpu().numpy()
    confidences = results[0].keypoints.conf[0].cpu().numpy()

    # WHY results[0].boxes.conf?
    # This is the overall "how confident is the
    # model that a person exists here at all"
    # score — separate from per-keypoint confidence.
    box_confidence = float(results[0].boxes.conf[0].cpu().numpy())

    return {
        'keypoints': keypoints,
        'confidences': confidences,
        'box_confidence': box_confidence,
    }


def compare(image_path, stock_model='yolov8n-pose.pt', finetuned_model='models/weights/best.pt'):
    print(f"Comparing models on: {image_path}\n")

    print(f"Running STOCK model ({stock_model})...")
    stock = run_detection(stock_model, image_path)

    print(f"Running FINE-TUNED model ({finetuned_model})...")
    finetuned = run_detection(finetuned_model, image_path)

    if stock is None or finetuned is None:
        print("\n One or both models failed to detect a person.")
        return

    print(f"\n{'Metric':<25s} {'Stock (nano)':>15s} {'Fine-tuned (small)':>20s} {'Difference':>12s}")

    # WHY compare box_confidence first?
    # This is the single clearest "did the model
    # get better" number. Higher = more confident
    # this is actually a person.
    diff = finetuned['box_confidence'] - stock['box_confidence']
    print(f"{'Detection confidence':<25s} {stock['box_confidence']:>15.3f} {finetuned['box_confidence']:>20.3f} {diff:>+12.3f}")

    print(f"\n{'Keypoint':<18s} {'Stock conf':>12s} {'Fine-tuned conf':>16s} {'Position shift (px)':>20s}")


    # WHY loop through all 17 keypoints?
    # A single overall number can hide important
    # detail — maybe the fine-tuned model is much
    # better at SHOULDERS specifically (which is
    # what my feature extraction relies on most)
    # but worse at ankles (which I don't even use).
    # Per-keypoint comparison shows exactly where
    # the improvement (or regression) is.
    for i, name in enumerate(LABELS):
        stock_conf = stock['confidences'][i]
        ft_conf = finetuned['confidences'][i]


        # WHY Euclidean distance for position shift?
        # Even if both models detect a keypoint
        # confidently, they might place it in
        # slightly different pixel locations.
        # This shows how much the predicted
        # position itself moved between models.
        position_shift = np.linalg.norm(
            finetuned['keypoints'][i] - stock['keypoints'][i]
        )

        # WHY mark improvements with an arrow?
        # Scanning a table of 17 rows of numbers
        # is tedious. A quick visual marker (↑/↓)
        # next to confidence let helps me to spot patterns
        # at a glance — e.g. "all upper body joints
        # improved, lower body got worse".

        marker = "↑" if ft_conf > stock_conf else ("↓" if ft_conf < stock_conf else "=")

        print(f"{name:<18s} {stock_conf:>12.3f} {ft_conf:>15.3f} {marker} {position_shift:>18.1f}")


    # WHY a summary at the end?
    # The per-keypoint table is detailed but dense.
    # A simple average gives me the headline
    # number to quote — "fine-tuned model improved
    # average keypoint confidence by X%".
    avg_stock_conf = np.mean(stock['confidences'])
    avg_ft_conf = np.mean(finetuned['confidences'])
    improved_count = sum(
        1 for i in range(17)
        if finetuned['confidences'][i] > stock['confidences'][i]
    )

    print(f"\nSUMMARY")
    print(f" Average keypoint confidence — stock: {avg_stock_conf:.3f}")
    print(f" Average keypoint confidence — fine-tuned: {avg_ft_conf:.3f}")
    print(f" Keypoints improve: {improved_count}/17")

    if avg_ft_conf > avg_stock_conf:
        pct_improvement = ((avg_ft_conf - avg_stock_conf) / avg_stock_conf) * 100
        print(f"\n Fine-tuned model shows {pct_improvement:.1f}% higher average confidence")
    else:
        pct_decline = ((avg_stock_conf - avg_ft_conf) / avg_stock_conf) * 100
        print(f"\n  Fine-tuned model shows {pct_decline:.1f}% lower average confidence")
        print("This can happen with only 10 epochs")
        print("   typically shows clearer improvement.")


if __name__ == "__main__":
    IMAGE_PATH = "data/samples/test.jpg"
    compare(IMAGE_PATH)