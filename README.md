# TellTrace — AI-Powered Body Language Reader

TellTrace analyses facial expressions, posture, and gestures from images and video to identify signals like confidence, stress, and engagement. It combines a fine-tuned YOLOv8 pose detection model, facial emotion classification, and Gemini's multimodal reasoning to interpret body language holistically rather than through fixed rule-based thresholds.

## What it does

Given an image or short video of a person, TellTrace:

1. Detects 17-point body pose (shoulders, elbows, wrists, hips, etc.) using a fine-tuned YOLOv8-pose model
2. Extracts structured posture features — spine tilt, arm openness, shoulder symmetry, forward lean, and more
3. Detects facial emotion using DeepFace
4. Sends the image and extracted features to Gemini, which reasons about confidence, stress, and engagement together rather than from rules alone
5. For video, samples frames at intervals and produces a timeline showing how signals change over time
6. Displays results through a FastAPI backend and browser-based frontend, including a chart visualisation for video timelines

## Tech stack

| Layer | Technology |
|---|---|
| Pose detection | YOLOv8 (Ultralytics), fine-tuned on COCO Keypoints |
| Emotion detection | DeepFace |
| Signal interpretation | Google Gemini API (`gemini-2.5-flash`) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, vanilla JavaScript |
| Visualisation | Matplotlib |
| Core libraries | Python, OpenCV, NumPy, PyTorch |

## Project structure

```
body-language-reader/
├── data/
│   ├── raw/                  # original images/videos you collect
│   ├── annotated/            # labeled datasets (YOLO format), if used
│   ├── samples/               # test images, test video, generated timeline + chart
│   │   └── frames/            # auto-extracted video frames (regenerated each run)
│   └── uploads/                # temporary storage for files uploaded via the web UI
│
├── models/
│   ├── weights/                # best.pt (fine-tuned), best.onnx (exported)
│   └── exports/
│
├── src/
│   ├── detection/
│   │   ├── pose.py             # YOLOv8 pose detection — hello world / standalone test
│   │   ├── emotion.py          # DeepFace emotion detection
│   │   └── compare_models.py   # stock vs fine-tuned model comparison tool
│   ├── features/
│   │   └── extract.py          # keypoints → posture feature extraction
│   ├── signals/
│   │   └── classifier.py       # Gemini-based multi-signal analysis
│   ├── video/
│   │   ├── process.py          # video frame sampling + timeline generation
│   │   └── visualise.py        # timeline → chart (matplotlib)
│   └── api/
│       ├── main.py             # FastAPI backend (/analyse-image, /analyse-video)
│       └── static/
│           └── index.html      # browser frontend
│
├── notebooks/                  # Colab/Jupyter training notebooks
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or 3.11 (newer versions may lag behind on ML library support)
- pip
- A free [Google AI Studio](https://aistudio.google.com) account for a Gemini API key
- Roughly 2GB of disk space for model weights and dependencies
- A webcam or sample images/video if you want to test with your own media

No GPU is required to run the project day-to-day — pose detection and emotion classification run fine on CPU. A GPU (or Google Colab) is only needed if you want to fine-tune the YOLOv8 model yourself.

## Setup — getting started from a fresh clone

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd body-language-reader
```

### 2. Create and activate a virtual environment

```bash
python -m venv yolo-env

# Windows
yolo-env\Scripts\activate

# macOS / Linux
source yolo-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you're on Windows and want GPU-accelerated PyTorch (NVIDIA GPUs only — not needed for AMD/Intel GPUs or CPU-only setups):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you don't have an NVIDIA GPU, the standard `torch` install from `requirements.txt` (CPU version) is all you need.

### 4. Get a free Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API key**
3. Copy the key

### 5. Create your `.env` file

Create a file named `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
```

This file is git-ignored and never committed — each person running this project needs their own key.

### 6. Add a test image

Place any clear photo of a person (yourself, a friend, anything with a visible upper body) into `data/samples/` and name it `test.jpg`. This is the default test image referenced throughout the codebase.

### 7. Verify the setup

Run the basic detection script to confirm everything is wired up correctly:

```bash
python -m src.detection.pose
```

You should see a JPG with a skeleton overlay saved to `data/samples/output_pose.jpg`, plus 17 keypoint coordinates printed to the terminal. If this works, your environment is ready.

## Usage

### Analyse a single image (command line)

```bash
python -m src.signals.classifier
```

This runs the full pipeline — pose detection, posture feature extraction, emotion detection, and Gemini analysis — on `data/samples/test.jpg`, printing a full report with confidence/stress/engagement scores, key observations, and a recommendation.

### Analyse a video (command line)

Place a short video (10-30 seconds works best for testing) at `data/samples/test_video.mp4`, then run:

```bash
python -m src.video.process
```

This samples one frame every 2 seconds, runs the full pipeline on each sampled frame, and saves a timeline to `data/samples/timeline.json`. A summary table prints to the terminal.

To visualise the timeline as a chart:

```bash
python -m src.video.visualise
```

This generates `data/samples/timeline_chart.png` showing confidence, stress, and engagement over time.

### Run the web app

Start the backend:

```bash
python -m uvicorn src.api.main:app --reload
```

Then open your browser to:

```
http://127.0.0.1:8000/app/
```

Upload an image or video directly through the browser interface to see results rendered visually, including signal bars, observations, and (for video) a results table.

You can also test the raw API directly via the interactive docs at `http://127.0.0.1:8000/docs`.

### Compare the stock vs fine-tuned pose model

If you've fine-tuned your own YOLOv8 model (see below), you can compare it against the stock pretrained model on the same image:

```bash
python -m src.detection.compare_models
```

This prints a per-keypoint confidence comparison and a summary of which model performs better, and by how much.

## Fine-tuning the pose model (optional)

The project ships with the stock `yolov8n-pose.pt` model by default, which works out of the box. If you want to fine-tune your own model on COCO Keypoints:

1. Open the training notebook in `notebooks/` using Google Colab (recommended — free GPU) or locally via Jupyter in VS Code (slower on CPU, but works)
2. Follow the cells in order — they handle dataset download, training configuration, and evaluation automatically
3. Once training completes, copy the resulting `best.pt` to `models/weights/best.pt`
4. Update the model path in `src/features/extract.py` and `src/signals/classifier.py`:

```python
# Change this:
model = YOLO('yolov8n-pose.pt')

# To this:
model = YOLO('models/weights/best.pt')
```

Training on a free Colab T4 GPU takes roughly 60 minutes for 100 epochs. Training locally on CPU is significantly slower (potentially several hours) and is only recommended for short test runs with a reduced epoch count.

## How the signals are computed

Posture features are derived directly from pose keypoint coordinates using geometry — angles, distances, and ratios between joints (for example, spine tilt is the angle between the shoulder midpoint and hip midpoint relative to vertical). These features, along with the detected facial emotion, are sent to Gemini together with the original image. Gemini reasons about all of these signals holistically — for example, distinguishing "arms crossed defensively" from "hands relaxed in pockets" using visual context that geometry alone cannot capture — and returns confidence, stress, and engagement scores between 0.0 and 1.0, along with natural-language observations and a recommendation.

Scores are validated and normalised in code after the API call, since even with a structured response schema, an LLM can occasionally return a percentage (e.g. 75) instead of a decimal (0.75). This is handled defensively rather than assumed away.

## Known limitations

- Designed for single-person analysis; multi-person scenes are not yet supported
- Lower-body keypoints (knees, ankles) are unreliable when the image only shows the upper body, which is the common case for this use case (interviews, desk setups, video calls)
- The free Gemini API tier allows 15 requests per minute, which limits video processing speed — long videos at fine sampling intervals will take proportionally longer
- Emotion detection requires a visible, unobstructed face; sunglasses or extreme angles reduce reliability

## Roadmap / possible next steps

- Train a custom emotion classifier (EfficientNet-B0 on AffectNet) to replace or supplement DeepFace
- Support multi-person detection and per-person signal tracking
- Add authentication and persistent storage for the web app
- Package the fine-tuned model as ONNX for faster, hardware-agnostic inference