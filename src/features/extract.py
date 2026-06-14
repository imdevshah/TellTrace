import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt')

def extract_features(kps):
    """
    Takes 17 keypoints (x, y) and returns a dict of
    body language features ready for signal classification.
    """
    nose          = kps[0]
    left_shoulder = kps[5];  right_shoulder = kps[6]
    left_elbow    = kps[7];  right_elbow    = kps[8]
    left_wrist    = kps[9];  right_wrist    = kps[10]
    left_hip      = kps[11]; right_hip      = kps[12]

    shoulder_mid   = (left_shoulder + right_shoulder) / 2
    hip_mid        = (left_hip      + right_hip)      / 2
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder) + 1e-6

    def angle_between(v1, v2):
        v1n = v1 / (np.linalg.norm(v1) + 1e-6)
        v2n = v2 / (np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(np.dot(v1n, v2n), -1, 1)))

    # 1. Spine tilt — how upright is the person (degrees from vertical)
    spine_tilt = angle_between(shoulder_mid - hip_mid, np.array([0, -1]))

    # 2. Forward lean — is the person leaning toward or away from camera
    forward_lean = (nose[0] - shoulder_mid[0]) / shoulder_width

    # 3. Shoulder symmetry — are shoulders level or one raised (tension)
    shoulder_sym = abs(left_shoulder[1] - right_shoulder[1]) / shoulder_width

    # 4. Shoulder elevation — shoulders raised toward ears (stress signal)
    neck_len           = abs(shoulder_mid[1] - nose[1])
    shoulder_elevation = shoulder_width / (neck_len + 1e-6)

    # 5. Arm openness — elbows spread away from body (confidence signal)
    left_arm_out  = (shoulder_mid[0] - left_elbow[0])  / shoulder_width
    right_arm_out = (right_elbow[0]  - shoulder_mid[0]) / shoulder_width
    arm_openness  = (left_arm_out + right_arm_out) / 2

    # 6. Wrist height — raised hands means active gesturing
    wrist_height_l   = (shoulder_mid[1] - left_wrist[1])  / shoulder_width
    wrist_height_r   = (shoulder_mid[1] - right_wrist[1]) / shoulder_width
    avg_wrist_height = (wrist_height_l + wrist_height_r) / 2

    # 7. Head tilt — head tilted left or right
    head_tilt = (nose[0] - shoulder_mid[0]) / shoulder_width

    return {
        'spine_tilt':          round(float(spine_tilt), 2),
        'forward_lean':        round(float(forward_lean), 3),
        'shoulder_symmetry':   round(float(shoulder_sym), 3),
        'shoulder_elevation':  round(float(shoulder_elevation), 3),
        'arm_openness':        round(float(arm_openness), 3),
        'avg_wrist_height':    round(float(avg_wrist_height), 3),
        'head_tilt':           round(float(head_tilt), 3),
    }


def interpret_signals(features):
    """
    Rule-based signal interpretation.
    Replace this later with your trained LSTM classifier.
    """
    confidence = sum([
        features['arm_openness']        >  0.3,   # open arms
        features['spine_tilt']          < 10.0,   # upright posture
        features['forward_lean']        >  0.1,   # leaning in
    ])
    stress = sum([
        features['shoulder_elevation']  >  1.5,   # raised shoulders
        features['shoulder_symmetry']   >  0.15,  # uneven shoulders
        features['arm_openness']        <  0.1,   # closed/tight arms
    ])
    engagement = sum([
        features['avg_wrist_height']    >  0.2,   # hands active
        features['forward_lean']        >  0.05,  # leaning forward
        features['spine_tilt']          < 15.0,   # not slouching
    ])
    return {
        'confidence':  round(confidence / 3, 2),
        'stress':      round(stress     / 3, 2),
        'engagement':  round(engagement / 3, 2),
    }


def run(image_path):
    results = model(image_path, conf=0.5, verbose=False)

    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        print("❌ No person detected.")
        return

    print(f"✅ Detected {len(results[0].keypoints.xy)} person(s)\n")

    for i, kps_tensor in enumerate(results[0].keypoints.xy):
        kps      = kps_tensor.cpu().numpy()
        features = extract_features(kps)
        signals  = interpret_signals(features)

        print(f"Person {i + 1}")
        print("=" * 45)

        print("  POSTURE FEATURES:")
        for k, v in features.items():
            print(f"    {k:25s}: {v}")

        print("\n  BODY LANGUAGE SIGNALS:")
        bars = {k: '█' * int(v * 10) + '░' * (10 - int(v * 10)) for k, v in signals.items()}
        print(f"    Confidence  {bars['confidence']}  {signals['confidence']}")
        print(f"    Stress      {bars['stress']}  {signals['stress']}")
        print(f"    Engagement  {bars['engagement']}  {signals['engagement']}")
        print()


if __name__ == '__main__':
    run('data/samples/test.jpg')