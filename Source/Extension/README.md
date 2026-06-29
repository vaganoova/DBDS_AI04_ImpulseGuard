# ImpulseGuard Chrome Extension (MVP)

A browser popup that checks whether a purchase is impulsive, using the trained
model served by the FastAPI backend.

```
Chrome popup (this folder)  →  HTTP  →  FastAPI backend  →  impulse_pipeline.pkl
```

## 1. Start the backend (required)

The extension needs the local API running. From the **project root**:

```bash
.venv/bin/python -m uvicorn Source.Backend.api:app --port 8000
```

Leave it running. Check it works by opening http://127.0.0.1:8000/docs
(FastAPI's interactive test page) or http://127.0.0.1:8000/ (health check).

## 2. Load the extension in Chrome

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select this folder: `Source/Extension`
5. The 🛒 ImpulseGuard icon appears in your toolbar (pin it for easy access)

## 3. Use it

1. On any shopping page, click the ImpulseGuard icon.
2. Enter the price (or click **Detect price from page** — best effort), then
   pick the category, how often you buy it, time spent deciding, and the two
   yes/no questions.
3. Click **Check purchase** → you get the graded level + confidence.
4. Optionally tap a feedback button to tell it the true level — this is saved
   to `Data/Real_User_Data/feedback.csv` and folded into the next training run.

## Notes

- The backend must be running, or the popup shows "Could not reach backend".
- `hour` is captured automatically; the other 6 features come from the form.
- This is a local/development setup (the API allows all origins). For a real
  deployment the backend would be hosted and CORS locked down.
