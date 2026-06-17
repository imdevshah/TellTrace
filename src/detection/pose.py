from ultralytics import YOLO
import cv2
import os

# Load pretrained YOLOv8 pose model (downloads ~22MB automatically)
model = YOLO('yolov8n-pose.pt')

# Path to my test image — put any photo of a person in data/samples/
IMAGE_PATH = 'data/samples/test.jpg'

if not os.path.exists(IMAGE_PATH):
    print(f" No image found at {IMAGE_PATH}")
    print(" Add any JPG photo of a person to data/samples/ and rename it test.jpg")
else:
    print(f"Running detection on: {IMAGE_PATH}")
    results = model(IMAGE_PATH, conf=0.5)

    # Save annotated image with skeleton drawn on it
    annotated = results[0].plot()
    output_path = 'data/samples/output_pose.jpg'
    cv2.imwrite(output_path, annotated)
    print(f" Saved: {output_path}")

    # Print all 17 keypoints for the first detected person
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kps = results[0].keypoints.xy[0].cpu().numpy()
        labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        print(f"\nDetected {len(results[0].keypoints.xy)} person(s)")
        print("\n17 Keypoints for person 1:")
        print("-" * 40)
        for name, pt in zip(labels, kps):
            print(f"  {name:20s}: x={pt[0]:.1f}, y={pt[1]:.1f}")
    else:
        print("No person detected. Try a clearer image.")