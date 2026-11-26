"""Open the local dashboard after the inference service is ready."""
import os
import threading
import time
import urllib.request
import webbrowser
import uvicorn

URL = "http://127.0.0.1:8000"
def open_when_ready():
    for _ in range(120):
        try:
            with urllib.request.urlopen(URL + "/health",timeout=1) as response:
                import json
                if json.load(response).get("ready"):
                    webbrowser.open(URL)
                    return
        except (OSError, ValueError):
            pass
        time.sleep(.5)

if __name__ == "__main__":
    os.environ.setdefault("NEUROSEG_LOAD_CANDIDATES","1")
    threading.Thread(target=open_when_ready,daemon=True).start()
    uvicorn.run("neuroseg.api:app",host="127.0.0.1",port=8000)
