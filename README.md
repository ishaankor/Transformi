# Discord Bot - Machine Learning & Data Visualization

## Features
- Linear regression graphing with random data, CSV files, or manual input
- Neural network training and visualization using TensorFlow
- Interactive Discord UI components
- File generation (graphs, model architectures)

## 🛠️ How It Works

The bot uses a **hybrid approach**:
1. **Discord Bot** runs in the main thread
2. **Flask Web Server** runs in a background thread
3. **Render** pings the web server to keep the service alive
4. **Health checks** ensure the bot stays online 24/7

### Dependencies:
- **discord.py** - Discord API wrapper
- **tensorflow** - Neural network training
- **matplotlib** - Graph generation
- **pandas/numpy** - Data processing
- **flask** - Web server for health checks
- **scikit-learn** - Machine learning utilities

### SSL Configuration:
- Automatic SSL certificate handling for macOS
- TensorFlow threading optimization to prevent mutex issues

### Performance Optimizations:
- Reduced dataset sizes for free tier compatibility
- Optimized batch sizes for memory efficiency
- Background threading for web server

## 🔒 Security & Best Practices
- Environment variables for all secrets
- `.gitignore` prevents accidental token commits
- Health check endpoints for monitoring
- Error handling and graceful failure recovery

## 📁 Project Structure
```
├── Discord Bot.py          # Main bot file
├── keep_alive.py           # Flask web server for health checks
├── requirements.txt        # Python dependencies
├── render.yaml            # Render deployment config
├── Procfile               # Heroku deployment config
├── runtime.txt            # Python version specification
├── Dockerfile             # Docker deployment config
├── .env.example           # Environment variables template
└── README.md              # This file
```
