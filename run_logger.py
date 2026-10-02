#!/usr/bin/env python3
"""
Structured logging module for each TTS run.
- Writes human-readable logs to logs/runs.log
- Writes machine-readable JSON Lines to logs/runs.jsonl
- Provides complete diagnostics: input parameters, synthesis metrics,
  chunking statistics, file sizes, audio duration, RTF, and detailed error tracebacks.
"""

import os
import json
import time
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
RUNS_LOG_PATH = LOGS_DIR / "runs.log"
RUNS_JSONL_PATH = LOGS_DIR / "runs.jsonl"


def ensure_logs_dir():
    """Ensure the logs directory exists."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    sec = int(round(seconds))
    hrs = sec // 3600
    mins = (sec % 3600) // 60
    rem_sec = sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{rem_sec:02d}"
    return f"{mins:02d}:{rem_sec:02d}"


class RunTracker:
    """Tracks and logs the lifecycle of a single speech synthesis run."""

    def __init__(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        message_id: Optional[int] = None,
        source_filename: str = "unknown",
        source_type: str = "text",
        raw_char_count: int = 0,
        voice_label: str = "Silero — Xenia 2",
    ):
        ensure_logs_dir()
        self.run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.start_time = datetime.now()
        self.start_monotonic = time.monotonic()
        self.end_time: Optional[datetime] = None
        self.elapsed_total_sec: float = 0.0

        # User and source data
        self.user_id = user_id
        self.chat_id = chat_id
        self.message_id = message_id
        self.source_filename = source_filename
        self.source_type = source_type
        self.raw_char_count = raw_char_count
        self.num_chunks: int = 0
        self.voice_label = voice_label

        # Synthesis metrics
        self.synthesis_duration_sec: float = 0.0
        self.audio_duration_sec: float = 0.0
        self.rtf: float = 0.0  # Real-Time Factor: synthesis_time / audio_duration

        # Output files
        self.mp3_path: Optional[str] = None
        self.mp3_size_bytes: int = 0
        self.mp3_size_mb: float = 0.0
        self.was_split: bool = False
        self.parts: List[Dict[str, Any]] = []

        # Status and errors
        self.status: str = "IN_PROGRESS"
        self.error_stage: Optional[str] = None
        self.error_type: Optional[str] = None
        self.error_message: Optional[str] = None
        self.error_traceback: Optional[str] = None

    def record_chunks(self, num_chunks: int):
        """Record the number of processed text chunks."""
        self.num_chunks = num_chunks

    def record_synthesis(
        self,
        synthesis_duration_sec: float,
        audio_duration_sec: float,
        sample_rate: int = 48000
    ):
        """Record neural network synthesis performance metrics."""
        self.synthesis_duration_sec = round(synthesis_duration_sec, 2)
        self.audio_duration_sec = round(audio_duration_sec, 2)
        if self.audio_duration_sec > 0:
            self.rtf = round(self.synthesis_duration_sec / self.audio_duration_sec, 3)

    def record_output(
        self,
        mp3_path: Path,
        parts: Optional[List[Path]] = None,
        get_duration_fn=None
    ):
        """Record final MP3 output files and split information."""
        self.mp3_path = str(mp3_path)
        if mp3_path.exists():
            self.mp3_size_bytes = mp3_path.stat().st_size
            self.mp3_size_mb = round(self.mp3_size_bytes / (1024 * 1024), 2)

        if parts and len(parts) > 1:
            self.was_split = True
            self.parts = []
            for idx, p in enumerate(parts, 1):
                part_size = p.stat().st_size if p.exists() else 0
                part_dur = 0.0
                if get_duration_fn and p.exists():
                    try:
                        part_dur = get_duration_fn(p)
                    except Exception:
                        pass
                self.parts.append({
                    "part_num": idx,
                    "total_parts": len(parts),
                    "filename": p.name,
                    "size_mb": round(part_size / (1024 * 1024), 2),
                    "duration_sec": round(part_dur, 1),
                    "duration_formatted": format_seconds(part_dur) if part_dur > 0 else "N/A"
                })

    def record_success(self):
        """Mark run as SUCCESS and finalize logs."""
        self.status = "SUCCESS"
        self._finalize()
        self._write_logs()

    def record_error(
        self,
        stage: str,
        error: Exception,
        tb_str: Optional[str] = None
    ):
        """Mark run as ERROR and persist diagnostics."""
        self.status = "ERROR"
        self.error_stage = stage
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.error_traceback = tb_str or traceback.format_exc()
        self._finalize()
        self._write_logs()

    def _finalize(self):
        """Calculate total elapsed execution time."""
        self.end_time = datetime.now()
        self.elapsed_total_sec = round(time.monotonic() - self.start_monotonic, 2)

    def _write_logs(self):
        """Write structured entries to runs.log and runs.jsonl."""
        try:
            # 1. Append to human-readable runs.log
            with open(RUNS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(self._format_text_log())

            # 2. Append to machine-readable runs.jsonl
            with open(RUNS_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[RunTracker Logger Error] Failed to write run log: {e}")

    def _format_text_log(self) -> str:
        """Format a human-readable visual block for runs.log."""
        lines = []
        sep = "=" * 80
        lines.append(sep)
        icon = "🟢" if self.status == "SUCCESS" else "🔴"
        lines.append(f"{icon} [{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}] RUN: {self.run_id}")
        lines.append(f"👤 User: ID {self.user_id} | Chat: {self.chat_id} | Message ID: {self.message_id}")
        lines.append(f"📄 Source: {self.source_filename} (Type: {self.source_type}, {self.raw_char_count:,} chars, {self.num_chunks} chunks)")
        lines.append(f"🗣 Voice: {self.voice_label}")
        lines.append("-" * 80)

        if self.audio_duration_sec > 0:
            dur_fmt = format_seconds(self.audio_duration_sec)
            lines.append(
                f"⏱ Neural Synthesis: {self.synthesis_duration_sec}s "
                f"(Audio Duration: {dur_fmt}, RTF: {self.rtf})"
            )

        if self.mp3_path:
            lines.append(f"💾 Output MP3: {self.mp3_path} ({self.mp3_size_mb} MB)")

        if self.was_split:
            lines.append(f"✂️ File > 48 MB (Telegram limit) — split into {len(self.parts)} part(s):")
            for p in self.parts:
                lines.append(
                    f"   • Part {p['part_num']}/{p['total_parts']}: {p['filename']} "
                    f"({p['size_mb']} MB, ~{p['duration_formatted']})"
                )

        if self.status == "SUCCESS":
            lines.append(f"🟢 STATUS: SUCCESS | Total Elapsed Time: {self.elapsed_total_sec}s")
        else:
            lines.append(f"🔴 STATUS: ERROR at stage [{self.error_stage}] | Total Elapsed Time: {self.elapsed_total_sec}s")
            lines.append(f"⚠️ Type: {self.error_type}")
            lines.append(f"💬 Description: {self.error_message}")
            if self.error_traceback:
                lines.append("Stack Trace:")
                for tb_line in self.error_traceback.strip().split("\n"):
                    lines.append(f"   {tb_line}")

        lines.append(sep + "\n\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return full dictionary representation for JSON serialization."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_total_sec": self.elapsed_total_sec,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "source_filename": self.source_filename,
            "source_type": self.source_type,
            "raw_char_count": self.raw_char_count,
            "num_chunks": self.num_chunks,
            "voice_label": self.voice_label,
            "synthesis_duration_sec": self.synthesis_duration_sec,
            "audio_duration_sec": self.audio_duration_sec,
            "real_time_factor": self.rtf,
            "mp3_path": self.mp3_path,
            "mp3_size_mb": self.mp3_size_mb,
            "was_split": self.was_split,
            "parts_count": len(self.parts) if self.was_split else 1,
            "parts": self.parts,
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback
        }
