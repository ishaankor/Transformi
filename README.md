# Discord Bot - Replit Deployment

This Discord bot includes machine learning features, data visualization, and interactive commands.

## Features
- Linear regression graphing with random data, CSV files, or manual input
- Neural network training and visualization
- Interactive Discord UI components
- File generation (graphs, model architectures)

## Replit Deployment Instructions

### 1. Upload to Replit
- Go to [replit.com](https://replit.com)
- Create a new Repl
- Choose "Import from GitHub" or upload your files directly
- Make sure all files are uploaded including:
  - `Discord Bot.py` (main bot file)
  - `main.py` (entry point)
  - `keep_alive.py` (keep bot running)
  - `.replit` (Replit configuration)
  - `replit.nix` (Nix environment setup)
  - `requirements.txt` (dependencies)
  - `.env.example` (environment variables template)

### 2. Set Environment Variables
In your Repl's "Secrets" tab (🔒 icon), add:
- Key: `DISCORD_BOT_TOKEN`
- Value: Your Discord bot token from [Discord Developer Portal](https://discord.com/developers/applications)

### 3. Install Dependencies
Replit should automatically install dependencies from `requirements.txt`. If not, run:
```bash
pip install -r requirements.txt
```

### 4. Run the Bot
- Click the "Run" button in Replit
- The bot will start and the keep-alive server will run on port 8080
- Your bot should come online in Discord

### 5. Keep It Running (Optional)
For the free tier, your Repl will sleep after inactivity. To keep it running:
- Use a service like UptimeRobot to ping your Repl's URL every 5 minutes
- Your Repl URL will be something like: `https://your-repl-name.your-username.repl.co`

## Bot Commands
- `/graph_linear_regression` - Create linear regression graphs
- `/create_neural_network` - Train and visualize neural networks

## Notes
- The bot uses matplotlib with 'Agg' backend for headless operation
- Generated files (graphs, model architectures) are temporary
- Make sure your Discord bot has proper permissions in your server

## Troubleshooting
- If packages fail to install, try running `pip install --upgrade pip` first
- Ensure your Discord bot token is correct and the bot is added to your server
- Check the console for any error messages during startup