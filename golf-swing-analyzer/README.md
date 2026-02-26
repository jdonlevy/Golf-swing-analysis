# Golf Swing Analyzer (Local MVP)

Local-only web app to upload a golf swing video and get quick posture/tempo cues.

## What it does
- Upload a video (MP4/MOV/AVI).
- Runs MediaPipe Pose on sampled frames.
- Computes simple metrics: head stability, spine angle change, shoulder tilt, backswing tempo.
- Returns tips and metrics in the UI.

## Setup
```bash
cd /Users/james.donlevy/Codex/golf-swing-analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p /Users/james.donlevy/Codex/golf-swing-analyzer/app/models
# Download the MediaPipe pose model and place it here:
# /Users/james.donlevy/Codex/golf-swing-analyzer/app/models/pose_landmarker_full.task
uvicorn app.main:app --reload --port 8001
```

Open `http://localhost:8001`.

## Notes
- Best results from a face-on camera view with full body in frame.
- This is a prototype; no club tracking yet.
- All processing is local to your machine.

## Next steps (optional)
- Add swing phase detection (address, top, impact) using heuristics + optical flow.
- Add club detection and shaft angle estimation.
- Save annotated frames for visual feedback.
