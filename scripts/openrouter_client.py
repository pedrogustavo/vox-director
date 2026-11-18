#!/usr/bin/env python3
"""
OpenRouter API client for motion-collage.

Dependency-free wrapper around OpenRouter's unified API (https://openrouter.ai/api/v1):
  - Image generation: POST /images (synchronous, returns b64_json)
  - Video generation: POST /videos + GET /videos/{id} (asynchronous)
  - Text-to-Speech:   POST /audio/speech (synchronous binary audio stream)
  - Chat / LLM:       POST /chat/completions

Env: OPENROUTER_API_KEY must be set (https://openrouter.ai/settings/keys).
"""
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import urllib.error
import urllib.request

BASE = "https://openrouter.ai/api/v1"
UA = "motion-collage/0.1 (+https://openrouter.ai)"


class OpenRouterError(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Get one at https://openrouter.ai/settings/keys"
        )
    return k


def _headers(json_body: bool = True) -> dict:
    h = {
        "Authorization": f"Bearer {_key()}",
        "User-Agent": UA,
        "X-Title": "motion-collage",
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise OpenRouterError(f"POST {path} -> {e.code}: {e.read().decode(errors='replace')[:400]}") from e


def _post_bytes(path: str, payload: dict, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise OpenRouterError(f"POST {path} -> {e.code}: {e.read().decode(errors='replace')[:400]}") from e


def _get(path: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(BASE + path, headers=_headers(json_body=False))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return json.loads(body)
        except ValueError:
            raise OpenRouterError(f"GET {path} -> {e.code}: {body[:400]}") from e


def generate_image(model: str, prompt: str, **params) -> str:
    """Call POST /images and save the returned b64_json to a temp file; return its path."""
    body = {"model": model, "prompt": prompt}
    if params.get("aspect_ratio"):
        body["aspect_ratio"] = params["aspect_ratio"]
    if params.get("size"):
        body["size"] = params["size"]

    resp = _post_json("/images", body, timeout=180)
    data = resp.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise OpenRouterError(f"POST /images returned no image data: {str(resp)[:300]}")

    raw = base64.b64decode(data[0]["b64_json"])
    mime = data[0].get("media_type", "image/png")
    ext = mimetypes.guess_extension(mime) or ".png"
    fd, tmp_path = tempfile.mkstemp(prefix="or_img_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return tmp_path


def submit_video(model: str, prompt: str, **params) -> str:
    """Submit an asynchronous video job to POST /videos; return job id."""
    body = {"model": model, "prompt": prompt}
    if params.get("image"):
        body["frame_images"] = [
            {
                "type": "image_url",
                "image_url": {"url": params["image"]},
                "frame_type": "first_frame",
            }
        ]
    for k in ("duration", "resolution", "aspect_ratio", "generate_audio", "seed"):
        if params.get(k) is not None:
            body[k] = params[k]

    resp = _post_json("/videos", body, timeout=90)
    jid = resp.get("id")
    if not jid:
        raise OpenRouterError(f"POST /videos returned no id: {str(resp)[:300]}")
    return jid


def video_status(job_id: str) -> dict:
    """Poll GET /videos/{job_id} and normalize to {status, output, error}."""
    d = _get(f"/videos/{job_id}")
    st = d.get("status", "pending")
    if st == "completed":
        urls = d.get("urls") or []
        out = urls[0] if urls else f"{BASE}/videos/{job_id}/content"
        return {"status": "completed", "output": out, "error": None}
    if st in ("failed", "expired"):
        err = d.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err or st)
        return {"status": "failed", "output": None, "error": msg}
    return {"status": "pending", "output": None, "error": None}


def generate_speech(model: str, text: str, voice: str = "alloy",
                    response_format: str = "mp3", speed: float = 1.0) -> str:
    """Call POST /audio/speech and save the audio stream to a temp file; return its path."""
    body = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    raw = _post_bytes("/audio/speech", body, timeout=120)
    fd, tmp_path = tempfile.mkstemp(prefix="or_tts_", suffix=f".{response_format}")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return tmp_path


def to_data_url(file_path: str) -> str:
    """Encode a local media file as a base64 data URL (no external upload host needed)."""
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "application/octet-stream"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def download(url_or_path: str, dest: str) -> str:
    """Copy a local temp file or download a remote OpenRouter asset via curl."""
    if os.path.exists(url_or_path):
        os.replace(url_or_path, dest)
        return dest
    cmd = ["/usr/bin/curl", "-sL", "--retry", "3", "-o", dest]
    if url_or_path.startswith(BASE):
        cmd += ["-H", f"Authorization: Bearer {_key()}"]
    cmd.append(url_or_path)
    subprocess.run(cmd, check=True)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise OpenRouterError(f"download produced empty file: {url_or_path}")
    return dest


def chat(model: str, messages: list, **params) -> str:
    """OpenAI-compatible chat completion -> assistant text."""
    body = {"model": model, "messages": messages, **params}
    resp = _post_json("/chat/completions", body, timeout=120)
    return resp["choices"][0]["message"]["content"]
