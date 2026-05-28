GrievX — Prediction Server
===========================

This describes how to run a minimal Flask server that accepts video uploads and returns a JSON prediction.

Setup
-----
1. Create and activate your virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run server (development):

```bash
python server.py
```

The server exposes `POST /predict` on port `8000` by default.

Examples
--------

Curl upload:

```bash
curl -F "file=@/path/to/video.mp4" http://localhost:8000/predict
```

Browser test:
Open `web_upload.html` in the repo root while the server is running (same origin assumed for simplicity). The page posts to `/predict` and shows JSON response.

Notes
-----
- Ensure you have a trained model at `runs/classify/grievx_yolo26/weights/best.pt` or edit `server.py` to point to your model.
- For production use, run with Gunicorn: `gunicorn -w 2 -b 0.0.0.0:8000 server:app` and put behind a reverse proxy.
- For large videos or many requests, use a background worker queue and persist uploads to object storage.
