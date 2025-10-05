from flask import Flask, jsonify
from threading import Thread
import time
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Discord Bot is running!",
        "message": "Bot is active and healthy",
        "timestamp": time.time()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "uptime": time.time(),
        "bot": "Discord ML Bot"
    })

@app.route('/ping')
def ping():
    return jsonify({"message": "pong", "status": "active"})

def run():
    # Use the PORT environment variable that Render provides
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

def keep_alive():
    t = Thread(target=run, daemon=True)
    t.start()
    print(f"Flask web server started for health checks")