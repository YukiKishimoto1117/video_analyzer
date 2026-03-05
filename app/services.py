import json
from pathlib import Path

import google.generativeai as genai
import whisper

from app.config import settings


_PROMPT = """
あなたは動画編集のアシスタントです。
次の文字起こしを、連続したコーナー単位に分割してください。
各コーナーについて、以下のJSON配列を返してください。
[
  {
    "start_sec": 0,
    "end_sec": 123.4,
    "title": "コーナータイトル",
    "summary": "コーナーの要約"
  }
]
制約:
- start_sec/end_secは秒数の数値
- 時系列順で重複なし
- JSON以外の文章は含めない
""".strip()


def transcribe_video(video_path: Path) -> dict:
    model = whisper.load_model(settings.whisper_model)
    return model.transcribe(str(video_path), language="ja")


def _fallback_segments(transcript: dict) -> list[dict]:
    text = transcript.get("text", "").strip()
    duration = 0.0
    segments = transcript.get("segments", [])
    if segments:
        duration = float(segments[-1].get("end", 0.0))

    return [
        {
            "start_sec": 0.0,
            "end_sec": max(duration, 1.0),
            "title": "全体",
            "summary": text[:400] if text else "文字起こし結果が空でした。",
        }
    ]


def segment_corners(transcript: dict) -> list[dict]:
    if not settings.gemini_api_key:
        return _fallback_segments(transcript)

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"{_PROMPT}\n\n文字起こし:\n{transcript}"
    response = model.generate_content(prompt)
    raw = (response.text or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return _fallback_segments(transcript)
