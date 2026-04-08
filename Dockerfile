FROM python:3.12-slim

# ffmpeg is required by whisper to decode audio formats (m4a, mp3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download the whisper model so it's baked into the image
RUN python -c "import whisper; whisper.load_model('base')"

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
