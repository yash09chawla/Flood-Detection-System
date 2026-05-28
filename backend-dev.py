"""
Local development launcher for the flood-prediction backend.

Starts uvicorn (the FastAPI server) and an ngrok tunnel so the Vercel
frontend can reach your local Python API.

Usage:
    python backend-dev.py

Environment:
    NGROK_AUTH_TOKEN     — your ngrok auth token (get from dashboard.ngrok.com)
    FLOOD_CHECKPOINT     — path to .pth checkpoint
    FLOOD_GEE_KEY        — path to GEE service-account JSON
    FLOOD_JOB_DIR        — base dir for job outputs

Once running, paste the printed https://...ngrok-free.app URL into
the Next.js frontend's NEXT_PUBLIC_API_BASE env var.
"""

import os
import sys
import threading
import time

# Force UTF-8 stdout/stderr so the Windows console (cp1252) doesn't choke on
# Unicode glyphs like → in our log messages.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PORT = int(os.environ.get("FLOOD_PORT", "8000"))


def _start_ngrok():
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("[dev] pyngrok not installed. Install with: pip install pyngrok")
        print(f"[dev] Or run ngrok manually:  ngrok http {PORT}")
        return

    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)
    else:
        print("[dev] WARNING: NGROK_AUTH_TOKEN not set. Free anonymous tunnel "
              "may have stricter limits.")

    tunnel = ngrok.connect(PORT, "http", bind_tls=True)
    public_url = tunnel.public_url
    print()
    print("=" * 70)
    print(f"  NGROK PUBLIC URL: {public_url}")
    print(f"  In frontend/.env.local set:")
    print(f"    NEXT_PUBLIC_API_BASE={public_url}")
    print("=" * 70)
    print()


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flood-detection-src"))

    # Launch ngrok in a background thread so it doesn't block uvicorn
    threading.Thread(target=_start_ngrok, daemon=True).start()

    # Small delay so the ngrok URL prints before uvicorn's startup logs
    time.sleep(1.0)

    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True,
                reload_dirs=[os.path.join(os.path.dirname(__file__), "flood-detection-src")])


if __name__ == "__main__":
    main()
