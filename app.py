# app.py - ClipForge AI backend (clean, Replit-friendly)

import os
import json
import shutil
import asyncio
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple
from zipfile import ZipFile

import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

# ==============================
# Basic App Config
# ==============================

APP_TITLE = "ClipForge AI Backend"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# OpenAI client (use env var)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY is not set. /transcribe and /ai_chat will fail.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==============================
# CORS (allow everything for now)
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost",
        "http://localhost",
        "capacitor://localhost",
        "ionic://localhost",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Storage Directories
# ==============================

BASE_DIR = os.path.join(os.getcwd(), "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PREVIEW_DIR = os.path.join(BASE_DIR, "previews")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
THUMB_DIR = os.path.join(BASE_DIR, "thumbs")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

for d in (UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR, THUMB_DIR, TMP_DIR):
    os.makedirs(d, exist_ok=True)

# Static mounts (video previews, exports, thumbnails)
app.mount("/media/previews", StaticFiles(directory=PREVIEW_DIR), name="previews")
app.mount("/media/exports", StaticFiles(directory=EXPORT_DIR), name="exports")
app.mount("/media/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")

# Base URL override (for production like Render / Replit)
PUBLIC_BASE = os.getenv("PUBLIC_BASE", "").rstrip("/")


def nowstamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def safe(name: str) -> str:
    return "".join(c for c in (name or "file") if c.isalnum() or c in ("-", "_", "."))[:120]


def run(cmd: List[str], timeout=1200) -> Tuple[int, str]:
    """Run a shell command, return (code, combined stdout+stderr)."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def scale_filter(h: int) -> str:
    return f"scale=-2:{h}:flags=lanczos"


def compose_vf(scale: Optional[str], drawtext: Optional[str]) -> List[str]:
    if scale and drawtext:
        return ["-vf", f"{scale},drawtext={drawtext}"]
    if scale:
        return ["-vf", scale]
    if drawtext:
        return ["-vf", f"drawtext={drawtext}"]
    return []


def drawtext_expr(text: str) -> str:
    t = (text or "").replace("'", r"\'")
    return (
        f"text='{t}':x=w-tw-20:y=h-th-20:"
        "fontcolor=white:fontsize=28:box=1:boxcolor=black@0.45:boxborderw=10"
    )


def hhmmss_to_seconds(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    parts = [float(p) for p in s.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(s)


def duration_from(start: str, end: str) -> float:
    return max(0.1, hhmmss_to_seconds(end) - hhmmss_to_seconds(start))


def seconds_to_text(x: float) -> str:
    x = max(0, int(round(x)))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def abs_url(request: Request, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = PUBLIC_BASE or str(request.base_url).rstrip("/")
    return f"{base}{path}"


def download_to_tmp(url: str) -> str:
    """
    Download a remote video to a temp file using yt-dlp.
    Returns the local filepath.
    """
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=TMP_DIR).name

    # You can also point cookies here if needed:
    cookies_file = os.path.join(BASE_DIR, "cookies.txt")
    ydl_opts = {
        "outtmpl": tmp_path,
        "format": "mp4/best",
        "noplaylist": True,
        "quiet": True,
    }
    if os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Add `yt-dlp` to requirements.txt")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(tmp_path):
        raise RuntimeError("yt-dlp did not produce a file.")
    return tmp_path


def make_thumbnail(source_path: str, t_start: str, out_path: str):
    """Grab a frame ~0.25s after start to avoid black frames."""
    seek = max(0.0, hhmmss_to_seconds(t_start) + 0.25)
    code, err = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(seek),
            "-i",
            source_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-1",
            "-y",
            out_path,
        ],
        timeout=30,
    )
    if code != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"thumbnail failed: {err[:300]}")


# ==============================
# Healthcheck
# ==============================

@app.get("/")
def health_root():
    return {"ok": True, "service": APP_TITLE, "version": APP_VERSION}


@app.get("/api/health")
def health_api():
    return {"ok": True}


# ==============================
# Core: Build a Single Clip
# ==============================

async def build_clip(
    source_path: str,
    start: str,
    end: str,
    want_preview: bool,
    want_final: bool,
    watermark_text: Optional[str],
) -> dict:
    base = safe(os.path.splitext(os.path.basename(source_path))[0])
    stamp = nowstamp()
    dur_s = duration_from(start, end)

    prev_name = f"{base}_{start.replace(':','-')}-{end.replace(':','-')}_prev_{stamp}.mp4"
    final_name = f"{base}_{start.replace(':','-')}-{end.replace(':','-')}_1080_{stamp}.mp4"
    prev_out = os.path.join(PREVIEW_DIR, prev_name)
    final_out = os.path.join(EXPORT_DIR, final_name)

    # PREVIEW (480p)
    if want_preview:
        if watermark_text:
            code, err = run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    start,
                    "-t",
                    str(dur_s),
                    "-i",
                    source_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "26",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    *compose_vf(scale_filter(480), drawtext_expr(watermark_text)),
                    "-movflags",
                    "+faststart",
                    "-y",
                    prev_out,
                ],
                timeout=900,
            )
            if code != 0 or not os.path.exists(prev_out):
                raise RuntimeError(f"Preview watermark failed: {err[:500]}")
        else:
            # try stream copy
            code, err = run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    start,
                    "-t",
                    str(dur_s),
                    "-i",
                    source_path,
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    "-y",
                    prev_out,
                ],
                timeout=300,
            )
            if code != 0 or not os.path.exists(prev_out):
                # fallback reencode
                code, err = run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        start,
                        "-t",
                        str(dur_s),
                        "-i",
                        source_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "28",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-movflags",
                        "+faststart",
                        "-y",
                        prev_out,
                    ],
                    timeout=600,
                )
                if code != 0 or not os.path.exists(prev_out):
                    raise RuntimeError(f"Preview failed: {err[:500]}")

    # FINAL (1080p)
    if want_final:
        code, err = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                start,
                "-t",
                str(dur_s),
                "-i",
                source_path,
                "-c:v",
                "libx264",
                "-preset",
                "faster",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                *compose_vf(
                    scale_filter(1080),
                    drawtext_expr(watermark_text) if watermark_text else None,
                ),
                "-movflags",
                "+faststart",
                "-y",
                final_out,
            ],
            timeout=1800,
        )
        if code != 0 or not os.path.exists(final_out):
            raise RuntimeError(f"Final export failed: {err[:500]}")

    # THUMBNAIL
    thumb_name = f"{base}_{start.replace(':','-')}_{stamp}.jpg"
    thumb_out = os.path.join(THUMB_DIR, thumb_name)
    try:
        make_thumbnail(source_path, start, thumb_out)
    except Exception:
        if os.path.exists(prev_out):
            try:
                make_thumbnail(prev_out, "00:00:00", thumb_out)
            except Exception:
                thumb_out = None
        else:
            thumb_out = None

    return {
        "preview_path": prev_out if os.path.exists(prev_out) else None,
        "final_path": final_out if os.path.exists(final_out) else None,
        "thumb_path": thumb_out if thumb_out and os.path.exists(thumb_out) else None,
        "duration_seconds": dur_s,
        "start": start,
        "end": end,
    }


# ==============================
# /clip_multi  (used by ClipForge)
# ==============================

@app.post("/clip_multi")
async def clip_multi(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
    sections: str = Form(...),
    watermark: str = Form("0"),
    wm_text: str = Form("@ClipForge"),
    preview_480: str = Form("1"),
    final_1080: str = Form("0"),
):
    tmp = None
    try:
        # Source input: file OR URL
        if file is not None:
            src = os.path.join(UPLOAD_DIR, safe(file.filename))
            with open(src, "wb") as f:
                f.write(await file.read())
        elif url:
            tmp = download_to_tmp(url)
            src = os.path.join(
                UPLOAD_DIR,
                safe(os.path.basename(url) or f"remote_{nowstamp()}.mp4"),
            )
            shutil.copy(tmp, src)
        else:
            return JSONResponse(
                {"ok": False, "error": "Provide a file or a url."}, status_code=400
            )

        # Parse sections JSON
        try:
            segs = json.loads(sections)
        except Exception:
            return JSONResponse(
                {"ok": False, "error": "sections must be valid JSON list"},
                status_code=400,
            )

        if not isinstance(segs, list) or not segs:
            return JSONResponse(
                {"ok": False, "error": "sections must be a non-empty list"},
                status_code=400,
            )

        wm = wm_text if watermark == "1" else None
        want_prev = preview_480 == "1"
        want_final = final_1080 == "1"

        sem = asyncio.Semaphore(3)

        async def worker(s, e):
            async with sem:
                r = await build_clip(src, s.strip(), e.strip(), want_prev, want_final, wm)
                return {
                    "start": s,
                    "end": e,
                    "duration_seconds": r["duration_seconds"],
                    "duration_text": seconds_to_text(r["duration_seconds"]),
                    "preview_url": abs_url(
                        request,
                        f"/media/previews/{os.path.basename(r['preview_path'])}",
                    )
                    if r["preview_path"]
                    else None,
                    "final_url": abs_url(
                        request, f"/media/exports/{os.path.basename(r['final_path'])}"
                    )
                    if r["final_path"]
                    else None,
                    "thumb_url": abs_url(
                        request, f"/media/thumbs/{os.path.basename(r['thumb_path'])}"
                    )
                    if r["thumb_path"]
                    else None,
                }

        tasks = [
            worker(str(s.get("start", "")), str(s.get("end", ""))) for s in segs
        ]
        results = await asyncio.gather(*tasks)

        # Optional: zip of finals
        zip_url = None
        if want_final:
            zip_name = f"clips_{nowstamp()}.zip"
            zip_path = os.path.join(EXPORT_DIR, zip_name)
            with ZipFile(zip_path, "w") as z:
                for r in results:
                    if r.get("final_url"):
                        fp = os.path.join(
                            EXPORT_DIR, os.path.basename(r["final_url"])
                        )
                        if os.path.exists(fp):
                            z.write(fp, arcname=os.path.basename(fp))
            zip_url = abs_url(request, f"/media/exports/{zip_name}")

        return JSONResponse({"ok": True, "items": results, "zip_url": zip_url})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ==============================
# /transcribe  (Whisper)
# ==============================

@app.post("/transcribe")
async def transcribe_audio(
    url: str = Form(None),
    file: UploadFile = File(None),
):
    if not client:
        return JSONResponse(
            {"ok": False, "error": "OPENAI_API_KEY missing"}, status_code=500
        )

    tmp_path = None
    audio_mp3 = None
    try:
        # 1) Input from file or URL
        if file:
            suffix = os.path.splitext(file.filename)[1] or ".webm"
            tmp_path = os.path.join(TMP_DIR, f"upl_{nowstamp()}{suffix}")
            with open(tmp_path, "wb") as f:
                f.write(await file.read())
        elif url:
            base = os.path.join(TMP_DIR, f"audio_{nowstamp()}")
            # Try yt-dlp audio
            code, err = run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "-x",
                    "--audio-format",
                    "mp3",
                    "--audio-quality",
                    "192K",
                    "-o",
                    base + ".%(ext)s",
                    "--force-overwrites",
                    url,
                ],
                timeout=900,
            )
            mp3_candidate = base + ".mp3"
            if code == 0 and os.path.exists(mp3_candidate):
                audio_mp3 = mp3_candidate
            else:
                # fallback: download full video then convert
                tmp_path = download_to_tmp(url)
        else:
            return JSONResponse(
                {"ok": False, "error": "No file or URL provided."}, status_code=400
            )

        # 2) Convert to mp3 if we don't have one yet
        if not audio_mp3:
            audio_mp3 = tmp_path.rsplit(".", 1)[0] + ".mp3"
            code, err = run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_path,
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    audio_mp3,
                ],
                timeout=900,
            )
            if code != 0 or not os.path.exists(audio_mp3):
                return JSONResponse(
                    {"ok": False, "error": f"FFmpeg audio convert failed: {err}."},
                    status_code=500,
                )

        # 3) Whisper transcription
        with open(audio_mp3, "rb") as a:
            tr = client.audio.transcriptions.create(
                model="whisper-1", file=a, response_format="text"
            )

        text_output = tr.strip() if isinstance(tr, str) else str(tr) or "(no text)"

        return JSONResponse({"ok": True, "text": text_output})
    except Exception as e:
        print("❌ /transcribe error:", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        for p in (tmp_path, audio_mp3):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ==============================
# /ai_chat   (ClipForge Assistant)
# ==============================

@app.post("/ai_chat")
async def ai_chat(request: Request):
    if not client:
        return {"ok": False, "error": "OPENAI_API_KEY missing"}

    form = await request.form()
    user_message = form.get("user_message", "") or ""
    transcript = form.get("transcript", "") or ""
    history_json = form.get("history", "[]") or "[]"

    try:
        history = json.loads(history_json)
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []

    messages = []

    # Add system context with transcript
    if transcript:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are ClipForge AI assistant. "
                    "You help with titles, hooks, summaries, best moments, and hashtags "
                    "based on this transcript:\n\n" + transcript
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are ClipForge AI assistant. You help with content ideas for video clips."
                ),
            }
        )

    # Add chat history
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    # Add latest user message
    if user_message:
        messages.append({"role": "user", "content": user_message})
    else:
        return {"ok": False, "error": "user_message is required"}

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    reply_text = completion.choices[0].message.content

    return {"ok": True, "reply": reply_text}


# ==============================
# /auto_clip  (Best 3 moments from transcript)
# ==============================

@app.post("/auto_clip")
async def auto_clip(transcript: str = Form(""), max_clips: str = Form("3")):
    if not client:
        return {"ok": False, "error": "OPENAI_API_KEY missing"}

    try:
        max_n = max(1, min(5, int(max_clips)))
    except Exception:
        max_n = 3

    system_msg = (
        "You are an editor for short-form content. "
        "Given a transcript, pick the most engaging short moments for clips. "
        "Return ONLY valid JSON: a list of objects with keys 'start', 'end', 'summary'. "
        "Times must be in 'HH:MM:SS' format. Example:\n"
        '[{"start":"00:00:10","end":"00:00:25","summary":"Funny story about failure"},'
        ' {"start":"00:01:05","end":"00:01:20","summary":"Biggest lesson about money"}]\n'
        f"Return at most {max_n} clips."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": transcript or "(no transcript)"},
        ],
    )

    raw = completion.choices[0].message.content

    clips = []
    try:
        # Find JSON in response
        start_idx = raw.find("[")
        end_idx = raw.rfind("]")
        if start_idx != -1 and end_idx != -1:
            json_str = raw[start_idx : end_idx + 1]
        else:
            json_str = raw
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            for c in parsed[:max_n]:
                clips.append(
                    {
                        "start": c.get("start", "00:00:00"),
                        "end": c.get("end", "00:00:10"),
                        "summary": c.get("summary", "").strip(),
                    }
                )
    except Exception:
        # fallback: no usable clips
        clips = []

    return {"ok": True, "clips": clips}


# ==============================
# /data-upload  (cookies.txt for yt-dlp)
# ==============================

@app.post("/data-upload")
async def data_upload(file: UploadFile = File(...)):
    try:
        dest_path = os.path.join(BASE_DIR, "cookies.txt")
        contents = await file.read()

        if not contents.strip():
            return {"ok": False, "error": "Uploaded file is empty"}

        with open(dest_path, "wb") as f:
            f.write(contents)

        # Validate first line (Netscape cookie file)
        with open(dest_path, "rb") as f:
            first_line = f.readline().decode(errors="ignore").strip()

        if "Netscape" not in first_line:
            return {
                "ok": False,
                "error": "Invalid cookies format. Must start with '# Netscape HTTP Cookie File'.",
            }

        return {"ok": True, "path": dest_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}
