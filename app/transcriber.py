from pathlib import Path

import whisper

_model = None


def _get_model(model_name: str = "small"):
    global _model
    if _model is None:
        _model = whisper.load_model(model_name)
    return _model


def transcribe_audio(file_path: str, language: str | None = None) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = _get_model()

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(path), **options)

    return {
        "text": result["text"].strip(),
        "language": result.get("language"),
        "duration": round(sum(s["end"] for s in result.get("segments", [])) or 0, 1) or None,
    }
