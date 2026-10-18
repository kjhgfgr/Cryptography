import os, signal, sys, time, hmac, hashlib
from flask import Flask, jsonify

app = Flask(__name__)
STARTED_AT = time.time()
ROCKET_SIZE = os.getenv("ROCKET_SIZE", "Small")
LAB_LOGIN = os.getenv("LAB_LOGIN") or "unknown"
LAB_TOKEN = os.getenv("LAB_TOKEN") or "unknown"

@app.get("/")
def index():
    return f"Hello from Docker! Rocket={ROCKET_SIZE}\n"

@app.get("/health")
def health():
    return jsonify(status="ok", uptime=round(time.time()-STARTED_AT,2))

@app.get("/proof")
def proof():
    secret = os.getenv("SECRET")
    payload = f"{LAB_LOGIN}:{LAB_TOKEN}".encode()
    if not secret:
        return jsonify(error="no-secret"), 400
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return jsonify(login=LAB_LOGIN, ok=True, proof=sig)

def handle_sigterm(*_):
    sys.stdout.write("Shutting down...\n"); sys.stdout.flush()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
