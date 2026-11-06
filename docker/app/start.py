"""Container entrypoint: start the prediction API, wait until the model is loaded, then start the website.

`docker stop` (SIGTERM) stops both cleanly. If either process dies, the container exits with an error so
Docker can restart it (`--restart unless-stopped`).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

API_PORT = int(os.getenv("API_PORT", "8000"))
APP_PORT = int(os.getenv("APP_PORT", "8501"))
procs = {}


def stop(code=0, *_):
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for p in procs.values():
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    sys.exit(code)


signal.signal(signal.SIGTERM, lambda *_: stop(0))
signal.signal(signal.SIGINT, lambda *_: stop(0))

procs["API"] = subprocess.Popen([sys.executable, "-m", "uvicorn", "churn_api:app", "--host", "0.0.0.0",
                                 "--port", str(API_PORT), "--no-server-header", "--log-level", "warning"])
for _ in range(240):
    if procs["API"].poll() is not None:
        print("The prediction API failed to start; see the messages above.", flush=True)
        stop(1)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=2) as r:
            if json.load(r).get("model_loaded"):
                break
    except OSError:
        pass
    time.sleep(0.5)
else:
    print("The prediction API did not load the model within 2 minutes.", flush=True)
    stop(1)

procs["website"] = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "churn_app.py", "--server.address", "0.0.0.0",
     "--server.port", str(APP_PORT)],
    env={**os.environ, "API_URL": f"http://127.0.0.1:{API_PORT}"})
print(f"\nChurn risk desk is running. Open http://localhost:{APP_PORT}  (API docs: http://localhost:{API_PORT}/docs"
      " if you published that port)\n", flush=True)

while True:
    for name, p in procs.items():
        if p.poll() is not None:
            print(f"The {name} stopped (exit code {p.returncode}); stopping the container.", flush=True)
            stop(1)
    time.sleep(1)
