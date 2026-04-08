FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy application code
COPY models.py /app/models.py
COPY server/ /app/server/
COPY openenv.yaml /app/openenv.yaml
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Expose port (HF Spaces default)
EXPOSE 7860

ENV PORT=7860
ENV PYTHONPATH=/app

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
