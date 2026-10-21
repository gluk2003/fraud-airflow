FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline pipeline
COPY services services
COPY data data
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
CMD ["python", "-m", "services.stream"]
