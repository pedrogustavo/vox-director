#!/usr/bin/env python3
"""
Manual / Google Flow prompt-pack & folder sync for motion-collage.

Let's you run the full motion-collage pipeline WITHOUT an Atlas Cloud API key:
  1. Export: builds all image, video, narration, and music prompts from beats.json
     using the exact same prompt composers as the automated stages, and writes:
       - <project>/prompts_flow.md + <project>/prompts_flow.json (images, clips, Flow Music BGM)
       - <project>/narration.txt (complete narration script for ElevenLabs / Google AI Studio;
         Flow Music only generates music/soundtracks, not speech)
  2. Sync (--sync): scans <project>/keyframes/, clips/, and audio/ for the
     files you generated, probes audio durations via ffprobe, updates beats.json,
     and reports what is ready vs. still missing before running assemble.py.

Usage:
  python3 scripts/manual_pack.py <project_dir>          # export prompt pack
  python3 scripts/manual_pack.py <project_dir> --sync   # sync local files -> beats.json
"""
import argparse
import json
import os
import subprocess
import sys

from clips import collage_prompt, painterly_prompt, VIDEO_MODEL as DEFAULT_VIDEO_MODEL
from keyframes import IMAGE_MODEL as DEFAULT_IMAGE_MODEL, shots_of
from styles import (
    compose_collage_prompt,
    compose_keyframe_prompt,
    resolve_theme,
    resolve_video_aspect,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

FLOW_MODEL_HINTS = {
    "google/nano-banana-2/text-to-image": "Nano Banana 2 (Image)",
    "openai/gpt-image-2/text-to-image": "GPT Image 2 (Image)",
    "google/gemini-omni-flash/image-to-video": "Gemini Omni Flash / Veo (Image-to-Video)",
    "kwaivgi/kling-video-o3-pro/image-to-video": "Kling O3 Pro (Image-to-Video — real people/brands)",
    "bytedance/seedance-2.0/reference-to-video": "Seedance 2.0 (Reference-to-Video)",
}


def _probe_dur(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
    ).stdout
    try:
        return round(float(out.strip()), 2)
    except ValueError:
        return 0.0


def _find_by_stem(directory: str, stem: str, exts: tuple[str, ...]) -> str | None:
    for ext in exts:
        candidate = os.path.join(directory, f"{stem}{ext}")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _ensure_dirs(project_dir: str) -> tuple[str, str, str]:
    kf_dir = os.path.join(project_dir, "keyframes")
    clip_dir = os.path.join(project_dir, "clips")
    audio_dir = os.path.join(project_dir, "audio")
    for d in (kf_dir, clip_dir, audio_dir):
        os.makedirs(d, exist_ok=True)
    return kf_dir, clip_dir, audio_dir


def _build_pack(doc: dict) -> dict:
    aspect = doc.get("aspect", "16:9")
    style = doc.get("style", "collage")
    img_model = doc.get("image_model", DEFAULT_IMAGE_MODEL)
    img_res = doc.get("image_resolution", "1k")
    vid_model = doc.get("video_model", DEFAULT_VIDEO_MODEL)
    vid_res = doc.get("video_resolution", "720p")

    theme = resolve_theme(doc.get("theme")) or {}
    collage_style = theme.get("idiom") or doc.get("collage_style", "american-retro")
    t_palette = theme.get("palette") or doc.get("palette")
    t_type = theme.get("type_style") or doc.get("type_style")
    t_finish = theme.get("finish") or doc.get("finish")
    era = doc.get("era")

    motion_style = doc.get("motion_style") or theme.get("motion_style", "punchy")
    constraints = doc.get("constraints", "strict")
    resolved_aspect, aspect_exact = resolve_video_aspect(aspect, vid_model)
    clip_aspect = resolved_aspect or aspect

    keyframes = []
    clips = []
    narrations = []

    for beat in doc.get("beats", []):
        bid = beat["id"]
        narrations.append({
            "beat_id": bid,
            "file": f"audio/narr_{bid}.mp3",
            "accepted_exts": list(AUDIO_EXTS),
            "language": doc.get("voice", {}).get("language", doc.get("language", "en")),
            "voice_id": doc.get("voice", {}).get("voice_id", "leo"),
            "speed": doc.get("voice", {}).get("speed", 1.0),
            "text": beat.get("narration", ""),
        })

        for shot, key in shots_of(beat):
            if shot.get("kind") == "host":
                continue

            if style == "collage":
                kf_prompt = shot.get("keyframe_prompt") or compose_collage_prompt(
                    shot["scene"],
                    beat.get("title_cn", ""),
                    beat.get("title_en", ""),
                    beat.get("bg", "warm ochre"),
                    aspect,
                    with_title=shot.get("title", True),
                    style=collage_style,
                    palette=t_palette,
                    type_style=t_type,
                    finish=t_finish,
                )
            else:
                kf_prompt = shot.get("keyframe_prompt") or compose_keyframe_prompt(
                    era, shot["scene"], beat.get("title_cn", ""), beat.get("title_en", ""), aspect
                )

            keyframes.append({
                "key": key,
                "beat_id": bid,
                "file": f"keyframes/kf_{key}.png",
                "alt_file": f"keyframes/kf_{key}.jpg",
                "model": img_model,
                "flow_hint": FLOW_MODEL_HINTS.get(img_model, img_model),
                "aspect": aspect,
                "resolution": img_res,
                "prompt": kf_prompt,
            })

            camera = shot.get("camera_move") or shot.get("motion", "push_in")
            element = shot.get("element_motion")
            if style == "collage":
                vid_prompt = collage_prompt(
                    camera,
                    element_motion=element,
                    feel=beat.get("feel"),
                    palette=beat.get("bg"),
                    amplitude=motion_style,
                    anchor_freeze=doc.get("anchor_freeze"),
                    has_title=shot.get("title", True),
                    constraints=constraints,
                )
            else:
                vid_prompt = painterly_prompt(shot.get("motion", camera))

            clips.append({
                "key": key,
                "beat_id": bid,
                "input_image": f"keyframes/kf_{key}.png (or .jpg)",
                "file": f"clips/clip_{key}.mp4",
                "model": vid_model,
                "flow_hint": FLOW_MODEL_HINTS.get(vid_model, vid_model),
                "aspect": clip_aspect,
                "aspect_exact": aspect_exact,
                "resolution": vid_res,
                "duration_s": int(shot.get("dur", 5)),
                "prompt": vid_prompt,
            })

    music_prompt = doc.get("music", "cinematic instrumental, no vocals")
    full_narration_text = "\n\n".join(
        nr["text"].strip() for nr in narrations if nr.get("text", "").strip()
    )
    voice_cfg = doc.get("voice", {})
    return {
        "project": doc.get("project", "vox-project"),
        "topic": doc.get("topic", ""),
        "aspect": aspect,
        "theme": doc.get("theme", collage_style),
        "keyframes": keyframes,
        "clips": clips,
        "narrations": narrations,
        "full_narration": {
            "text_file": "narration.txt",
            "audio_file": "audio/narration.mp3",
            "accepted_exts": list(AUDIO_EXTS),
            "language": voice_cfg.get("language", doc.get("language", "en")),
            "voice_id": voice_cfg.get("voice_id", "leo"),
            "speed": voice_cfg.get("speed", 1.0),
            "tools": "ElevenLabs or Google AI Studio (Flow Music generates music only, not speech)",
            "text": full_narration_text,
        },
        "bgm": {
            "file": "audio/bgm.mp3",
            "accepted_exts": list(AUDIO_EXTS),
            "model": "Google Flow Music (soundtrack/instrumental only — no narration)",
            "prompt": music_prompt,
        },
    }


def _render_markdown(pack: dict, project_dir: str) -> str:
    lines = [
        f"# Prompt Pack (Manual / Google Flow) — {pack['project']}",
        "",
        f"- **Topic:** {pack['topic']}",
        f"- **Aspect:** `{pack['aspect']}` | **Theme:** `{pack['theme']}`",
        f"- **Project folder:** `{project_dir}`",
        "",
        "Save all generated files into the exact subfolders below (`keyframes/`, `clips/`, `audio/`).",
        "When ready, run:",
        "```bash",
        f"python3 scripts/manual_pack.py {project_dir} --sync",
        f"python3 scripts/assemble.py {project_dir}",
        "```",
        "",
        "---",
        "",
        f"## Batch 1 — Keyframes ({len(pack['keyframes'])} images)",
        "",
        "| Shot | Save as | Model | Aspect | Res |",
        "|---|---|---|---|---|",
    ]
    for kf in pack["keyframes"]:
        lines.append(
            f"| `{kf['key']}` | `{kf['file']}` (or `.jpg`) | {kf['flow_hint']} | `{kf['aspect']}` | `{kf['resolution']}` |"
        )
    lines.append("")
    for idx, kf in enumerate(pack["keyframes"], 1):
        lines += [
            f"### Keyframe {idx}: `{kf['key']}` → `{kf['file']}`",
            f"- **Model:** `{kf['model']}` ({kf['flow_hint']})",
            f"- **Aspect / Resolution:** `{kf['aspect']}` · `{kf['resolution']}`",
            "```text",
            kf["prompt"],
            "```",
            "",
        ]

    lines += [
        "---",
        "",
        f"## Batch 2 — Video Clips ({len(pack['clips'])} clips, Image-to-Video)",
        "",
        "| Shot | Input image | Save as | Model | Aspect | Dur |",
        "|---|---|---|---|---|---|",
    ]
    for cl in pack["clips"]:
        note = "" if cl["aspect_exact"] else " (approx)"
        lines.append(
            f"| `{cl['key']}` | `{cl['input_image']}` | `{cl['file']}` | {cl['flow_hint']} | `{cl['aspect']}`{note} | {cl['duration_s']}s |"
        )
    lines.append("")
    for idx, cl in enumerate(pack["clips"], 1):
        lines += [
            f"### Clip {idx}: `{cl['key']}` → `{cl['file']}`",
            f"- **Input image:** `{cl['input_image']}`",
            f"- **Model:** `{cl['model']}` ({cl['flow_hint']})",
            f"- **Aspect / Duration / Res:** `{cl['aspect']}` · `{cl['duration_s']}s` · `{cl['resolution']}`",
            "```text",
            cl["prompt"],
            "```",
            "",
        ]

    bgm = pack["bgm"]
    fn = pack["full_narration"]
    lines += [
        "---",
        "",
        "## Batch 3 — Audio (Flow Music for Soundtrack + External TTS for Narration)",
        "",
        f"### 1. Background Music (Soundtrack) → `{bgm['file']}`",
        f"- **Tool / Model:** {bgm['model']}",
        "- *Note: Google Flow Music only generates music/soundtracks, not voice-over narration.*",
        "```text",
        bgm["prompt"],
        "```",
        "",
        f"### 2. Full Narration (Voice-Over) → `{fn['audio_file']}`",
        f"- **Script file:** `{fn['text_file']}` (ready to copy-paste)",
        f"- **Recommended tools:** {fn['tools']}",
        f"- **Language / Voice / Speed:** `{fn['language']}` · `{fn['voice_id']}` · `{fn['speed']}x`",
        "- *Save the exported audio as `audio/narration.mp3` (or `.wav`/`.m4a`). Per-beat files (`audio/narr_1.mp3`, ...) are also supported by `--sync`.*",
        "```text",
        fn["text"],
        "```",
        "",
    ]
    return "\n".join(lines)


def export_pack(project_dir: str) -> None:
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    _ensure_dirs(project_dir)
    pack = _build_pack(doc)

    md_path = os.path.join(project_dir, "prompts_flow.md")
    json_path = os.path.join(project_dir, "prompts_flow.json")
    narr_txt_path = os.path.join(project_dir, "narration.txt")
    with open(md_path, "w") as f:
        f.write(_render_markdown(pack, project_dir))
    with open(json_path, "w") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    with open(narr_txt_path, "w") as f:
        f.write(pack["full_narration"]["text"] + "\n")

    print(f"Exported prompt pack:")
    print(f"  Markdown:  {md_path}")
    print(f"  JSON:      {json_path}")
    print(f"  Narration: {narr_txt_path}")
    print(
        f"  Totals:    {len(pack['keyframes'])} keyframes, "
        f"{len(pack['clips'])} clips, 1 full narration ({len(pack['narrations'])} beats) + 1 bgm"
    )


def sync_files(project_dir: str) -> None:
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    kf_dir, clip_dir, audio_dir = _ensure_dirs(project_dir)

    missing_kf, missing_clips, missing_narr = [], [], []
    found_kf = found_clips = found_narr = 0

    for beat in doc.get("beats", []):
        bid = beat["id"]
        narr = _find_by_stem(audio_dir, f"narr_{bid}", AUDIO_EXTS)
        if narr:
            beat["narration_audio"] = narr
            beat["narration_dur"] = _probe_dur(narr)
            found_narr += 1
        else:
            missing_narr.append(f"audio/narr_{bid}.mp3")

        for shot, key in shots_of(beat):
            if shot.get("kind") == "host":
                continue
            kf = _find_by_stem(kf_dir, f"kf_{key}", IMAGE_EXTS)
            if kf:
                shot["keyframe_path"] = kf
                found_kf += 1
            else:
                missing_kf.append(f"keyframes/kf_{key}.png")

            clip = _find_by_stem(clip_dir, f"clip_{key}", VIDEO_EXTS)
            if clip:
                shot["clip_path"] = clip
                found_clips += 1
            else:
                missing_clips.append(f"clips/clip_{key}.mp4")

    full_narr = _find_by_stem(audio_dir, "narration", AUDIO_EXTS)
    total_beats = len(doc.get("beats", []))
    use_full_narr = False

    if found_narr == total_beats and total_beats > 0:
        doc.pop("narration_audio", None)
        doc.pop("narration_dur", None)
        narr_status = f"{found_narr}/{total_beats} per-beat files found"
    elif full_narr:
        use_full_narr = True
        full_dur = _probe_dur(full_narr)
        doc["narration_audio"] = full_narr
        doc["narration_dur"] = full_dur
        missing_narr = []
        narr_status = f"OK ({os.path.basename(full_narr)}, {full_dur}s — full track)"
    else:
        missing_narr = ["audio/narration.mp3 (or per-beat audio/narr_<id>.mp3)"]
        narr_status = f"MISSING (expected audio/narration.mp3 or {total_beats} per-beat files)"

    bgm = _find_by_stem(audio_dir, "bgm", AUDIO_EXTS)
    if bgm:
        doc["bgm_path"] = bgm
        doc["bgm_dur"] = _probe_dur(bgm)

    with open(bpath, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"Synced {bpath}:")
    print(f"  Keyframes:  {found_kf} found, {len(missing_kf)} missing")
    print(f"  Clips:      {found_clips} found, {len(missing_clips)} missing")
    print(f"  Narration:  {narr_status}")
    print(f"  BGM:        {'OK (' + os.path.basename(bgm) + ')' if bgm else 'MISSING (audio/bgm.mp3)'}")

    missing = missing_kf + missing_clips + ([] if use_full_narr else missing_narr) + ([] if bgm else ["audio/bgm.mp3"])
    if missing:
        print("\nStill missing:")
        for item in missing:
            print(f"  - {item}")
    else:
        print(f"\nAll assets present! Run: python3 scripts/assemble.py {project_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manual/Google Flow prompts or sync local assets.")
    parser.add_argument("project_dir", help="Project directory containing beats.json (e.g. out/my-video)")
    parser.add_argument("--sync", action="store_true", help="Scan keyframes/, clips/, audio/ and update beats.json")
    args = parser.parse_args()

    proj = os.path.abspath(args.project_dir)
    if args.sync:
        sync_files(proj)
    else:
        export_pack(proj)


if __name__ == "__main__":
    main()
