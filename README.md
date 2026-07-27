<p align="center">
  <img width="150" height="150" src="./public/transformi_logo.png" alt="logo">
</p>

> A conversational machine learning Discord bot that transforms various inputs into interactive models, visualizations, and data-driven insights.

Transformi positions itself as a data science proof-of-concept: a Discord-based analytic co-pilot that blends model experimentation, visualization, and deployment-ready application design.

---

### The Vision
Modern analytics often live in fragmented tools: notebooks, BI dashboards, and command-line scripts. Transformi brings this experience into a single, conversational interface. The project demonstrates how a data scientist can interact with a dataset through chat: generating regression plots, selecting features, training models, and receiving visual artifacts without leaving Discord.

### Engineering Highlights
- **Conversational ML integration:** Discord commands and rich UI components enable dynamic dataset selection, manual input, and on-demand model execution.
- **End-to-end modeling:** Supports linear regression, classification, neural network training, and ensemble methods using `scikit-learn` and `tensorflow`.
- **Automated visualization:** Generates charts and model outputs as image attachments, allowing users to interpret results immediately.
- **Deployment-friendly health layer:** A lightweight Flask service provides `/`, `/health`, and `/ping` endpoints to support managed hosting and uptime monitoring.
- **Context-aware interaction state:** User session locking and cached state management help the bot coordinate multi-step analytical workflows safely.
- **Robust data handling:** Uses `pandas`, `numpy`, and `pyarrow` for rapid preprocessing, dataset transformation, and CSV support.

### Technical Architecture
- `pandas`, `numpy`, `pyarrow` for data ingestion and transformation
- `scikit-learn` for regression, classification, preprocessing, and evaluation
- `tensorflow-cpu` for neural network training and model visualization
- `matplotlib`, `seaborn` for charts and analysis plots

### What Transformi Demonstrates
- Rapid exploration of raw and structured data through chat-driven inputs
- Visual regression analysis from generated or user-provided values
- Neural network training with immediate feedback and rendered artifacts
- Hybrid application design that combines bot UX with service monitoring

---