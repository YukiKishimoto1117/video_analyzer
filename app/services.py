import json
from pathlib import Path

import google.generativeai as genai
import whisper

from app.config import settings


_PROMPT = """
あなたは動画編集のアシスタントです。
入力として、Whisperのセグメント一覧（start/end/text）と、境界候補秒数を渡します。
境界候補を優先しつつ、文脈上明らかに不自然な場合のみ調整して、連続したコーナーに分割してください。
各コーナーについて、以下のJSON配列のみ返してください。
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
- 時系列順で重複なし、start_sec < end_sec
- 出力範囲は入力セグメントの範囲内
- JSON以外の文章は含めない
""".strip()


def transcribe_video(video_path: Path) -> dict:
    model = whisper.load_model(settings.whisper_model)
    language = None if settings.whisper_language.lower() == "auto" else settings.whisper_language
    return model.transcribe(
        str(video_path),
        language=language,
        beam_size=settings.whisper_beam_size,
        best_of=settings.whisper_best_of,
        temperature=settings.whisper_temperature,
    )


def _segment_rows(transcript: dict) -> list[dict]:
    rows: list[dict] = []
    for seg in transcript.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            continue
        rows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "text": (seg.get("text") or "").strip(),
            }
        )
    return rows


def _boundary_candidates(rows: list[dict]) -> list[float]:
    if not rows:
        return [0.0]

    candidates = [rows[0]["start_sec"]]
    for prev, cur in zip(rows, rows[1:]):
        gap = cur["start_sec"] - prev["end_sec"]
        if gap >= settings.merge_gap_sec:
            candidates.append(cur["start_sec"])

    candidates.append(rows[-1]["end_sec"])
    return sorted({round(x, 1) for x in candidates})


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


def _normalize_corners(corners: list[dict], rows: list[dict]) -> list[dict]:
    if not rows:
        return corners

    min_t = rows[0]["start_sec"]
    max_t = rows[-1]["end_sec"]

    cleaned: list[dict] = []
    for c in corners:
        try:
            start = max(min_t, float(c["start_sec"]))
            end = min(max_t, float(c["end_sec"]))
            if end <= start:
                continue
            cleaned.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "title": str(c.get("title", "コーナー")).strip() or "コーナー",
                    "summary": str(c.get("summary", "")).strip() or "要約なし",
                }
            )
        except (TypeError, ValueError, KeyError):
            continue

    cleaned.sort(key=lambda x: x["start_sec"])
    if not cleaned:
        return _fallback_segments({"segments": rows, "text": ""})

    merged: list[dict] = [cleaned[0]]
    for cur in cleaned[1:]:
        prev = merged[-1]
        if cur["start_sec"] - prev["end_sec"] <= settings.merge_gap_sec:
            if prev["end_sec"] - prev["start_sec"] < settings.min_corner_sec:
                prev["end_sec"] = max(prev["end_sec"], cur["end_sec"])
                prev["summary"] = f"{prev['summary']} {cur['summary']}".strip()
                continue
        merged.append(cur)

    merged[0]["start_sec"] = min_t
    merged[-1]["end_sec"] = max_t
    return merged


def segment_corners(transcript: dict) -> list[dict]:
    rows = _segment_rows(transcript)
    if not settings.gemini_api_key:
        return _fallback_segments(transcript)

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)

    payload = {
        "segment_rows": rows,
        "boundary_candidates_sec": _boundary_candidates(rows),
    }
    prompt = f"{_PROMPT}\n\n入力データ(JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    try:
        response = model.generate_content(prompt)
    except Exception:
        return _fallback_segments(transcript)

    raw = (response.text or "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.removeprefix("```json").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return _fallback_segments(transcript)

    return _normalize_corners(parsed, rows)
