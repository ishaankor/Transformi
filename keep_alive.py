from flask import Flask
from threading import Thread
import time

app = Flask('')

@app.route('/')
def home():
    return "Discord Bot is alive!"

@app.route('/health')
def health():
    return {"status": "healthy", "timestamp": time.time()}

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()