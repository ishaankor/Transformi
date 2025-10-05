# Discord Bot - Machine Learning & Data Visualization

This Discord bot includes machine learning features, data visualization, and interactive commands.

## Features
- Linear regression graphing with random data, CSV files, or manual input
- Neural network training and visualization using TensorFlow
- Interactive Discord UI components
- File generation (graphs, model architectures)

## 🚀 Deployment on Render (Recommended)

### Why Render Web Service?
- ✅ **Free tier available** for web services
- ✅ **Automatic health checks** keep the bot alive
- ✅ **Easy GitHub integration** for auto-deploys
- ✅ **Built-in SSL** and custom domains
- ✅ **Better reliability** than background workers

### Quick Deploy Steps:

1. **Fork/Clone this repository** to your GitHub account

2. **Go to [render.com](https://render.com)** and create an account

3. **Create New Web Service:**
   - Connect your GitHub account
   - Select this repository
   - Render will automatically detect `render.yaml` configuration

4. **Set Environment Variables:**
   - `DISCORD_BOT_TOKEN`: Your bot token from [Discord Developer Portal](https://discord.com/developers/applications)

5. **Deploy!** 
   - Render will build and deploy automatically
   - Your bot will be available at a URL like: `https://your-app-name.onrender.com`
   - The Flask web server provides health endpoints for Render to ping

### Health Check Endpoints:
- `GET /` - Main status page
- `GET /health` - Health check for Render
- `GET /ping` - Simple ping endpoint

## 🛠️ How It Works

The bot uses a **hybrid approach**:
1. **Discord Bot** runs in the main thread
2. **Flask Web Server** runs in a background thread
3. **Render** pings the web server to keep the service alive
4. **Health checks** ensure the bot stays online 24/7

## 📋 Bot Commands
- `/graph_linear_regression` - Create linear regression graphs with various data sources
- `/create_neural_network` - Train and visualize neural networks on MNIST data

## 🔧 Local Development

### Setup:
```bash
# Clone repository
git clone https://github.com/ishaankor/Transformi.git
cd Transformi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "DISCORD_BOT_TOKEN=your_token_here" > .env

# Run bot
python "Discord Bot.py"
```

## 🎯 Alternative Deployments

### Railway:
- Uses `railway.json` configuration
- Good free tier with $5 monthly credit

### Fly.io:
- Uses `Dockerfile` and `fly.toml`
- Docker-based deployment

### Heroku:
- Uses `Procfile` and `runtime.txt`
- Traditional PaaS deployment

## 🛠️ Technical Details

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

## � Troubleshooting

### Bot won't start:
- Check `DISCORD_BOT_TOKEN` is correctly set
- Verify bot has proper permissions in Discord server
- Check Render logs for specific error messages

### SSL Issues (macOS):
```bash
/Applications/Python\ 3.*/Install\ Certificates.command
```

### Memory Issues:
- Neural network uses reduced MNIST dataset (5000 training samples)
- Optimized for free tier resource limits

### Health Check Failing:
- Verify Flask server starts correctly
- Check if PORT environment variable is set
- Ensure `/health` endpoint responds correctly