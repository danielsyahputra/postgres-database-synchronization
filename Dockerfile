FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fr_sync_simulation.py .

CMD ["python", "fr_sync_simulation.py"]
