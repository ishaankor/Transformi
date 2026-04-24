# Use Python 3.10 base image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "Discord Bot.py"]