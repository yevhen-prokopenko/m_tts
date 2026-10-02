#!/usr/bin/env python3
"""
Telegram bot for automated local text-to-speech synthesis (Silero Xenia / Edge-TTS).
- Supports plain text, forwarded messages, .txt, .md, and .pdf documents.
- Normalizes incoming data into 01_Inbox/.
- Manages up to 3 concurrent tasks with an automatic queuing system.
- Live dynamic Telegram progress bar with flood-limit protection.
- Delivers optimized speech MP3 back to the user.
- Access control based on Telegram User ID whitelist.
"""

import os
import re
import html
import time
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Dict

from dotenv import load_dotenv
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, FSInputFile, BufferedInputFile, Document,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.filters import CommandStart, Command
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

import process
from process import (
    BASE_DIR, INBOX_DIR, RESULTS_DIR, DELETE_DIR,
    DEFAULT_VOICE, DEFAULT_UK_VOICE, DEFAULT_EN_VOICE, DEFAULT_PL_VOICE
)
from run_logger import RunTracker, RUNS_LOG_PATH, RUNS_JSONL_PATH
from translator import detect_language, translate_text, get_language_display_name

# Store for tasks awaiting language choice (job_id -> dict)
PENDING_JOBS = {}

# Buffer for merging split Telegram text chunks (chat_id -> dict)
TEXT_BUFFERS: Dict[int, dict] = {}
TEXT_BUFFER_DELAY = 2.0

# Load environment configuration
load_dotenv(BASE_DIR / ".env")
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ALLOWED_RAW = os.getenv("ALLOWED_USER_IDS", "").strip()

# Parse allowed numeric Telegram User IDs
ALLOWED_USER_IDS: Set[int] = set()
if ALLOWED_RAW:
    for uid_str in ALLOWED_RAW.split(","):
        uid_str = uid_str.strip()
        if uid_str.isdigit():
            ALLOWED_USER_IDS.add(int(uid_str))

# Semaphore: maximum 3 concurrent speech synthesis jobs
MAX_CONCURRENT_TASKS = 3
TASK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Task counters
ACTIVE_TASKS = 0
QUEUE_COUNTER = 0

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN) if BOT_TOKEN and BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN_HERE" else None
dp = Dispatcher()


def is_user_allowed(user_id: int) -> bool:
    """Check if the given Telegram user is authorized."""
    if not ALLOWED_USER_IDS:
        return False
    return user_id in ALLOWED_USER_IDS


def sanitize_filename(text: str, max_chars: int = 20) -> str:
    """Generate a sanitized filename from the first characters of text + timestamp."""
    first_line = text.strip().split("\n")[0].strip()
    first_clean = re.sub(r"[^\w\dа-яА-ЯёЁa-zA-Z]+", "_", first_line[:max_chars]).strip("_")
    if not first_clean:
        first_clean = "message"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{first_clean}_{timestamp}"


class TelegramProgressUpdater:
    """Sends and updates a dynamic progress bar in Telegram with flood-wait protection."""

    def __init__(self, bot_instance: Bot, chat_id: int, message_id: int, title: str, voice_label: str = DEFAULT_VOICE['label']):
        self.bot = bot_instance
        self.chat_id = chat_id
        self.message_id = message_id
        self.title = title
        self.voice_label = voice_label
        self.last_update_time = 0.0
        self.last_percent = -1

    async def update(self, percent: int, chunk_idx: int, total_chunks: int, stage: str = "Synthesis"):
        now = time.monotonic()
        # Rate limit updates: at most once every 1.5 seconds unless reaching 100%
        if (now - self.last_update_time < 1.5) and percent < 100 and percent == self.last_percent:
            return

        self.last_update_time = now
        self.last_percent = percent

        filled = max(0, min(10, int(percent / 10)))
        bar = "█" * filled + "░" * (10 - filled)
        text = (
            f"🎙 <b>{self.title}</b>\n\n"
            f"⚙️ {stage}: <code>[{bar}] {percent}%</code>\n"
            f"📊 Chunk: {chunk_idx}/{total_chunks}\n"
            f"🗣 Voice: {self.voice_label}"
        )
        try:
            await self.bot.edit_message_text(
                text=text,
                chat_id=self.chat_id,
                message_id=self.message_id,
                parse_mode="HTML"
            )
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except TelegramBadRequest:
            pass
        except Exception:
            pass


async def handle_tts_job(
    source_file_path: Path,
    chat_id: int,
    display_title: str,
    original_message_id: Optional[int] = None,
    user_id: Optional[int] = None,
    source_type: str = "text",
    raw_char_count: int = 0,
    voice_conf: dict = DEFAULT_VOICE
):
    """
    Process an individual file: queue with semaphore, render status,
    synthesize speech (Silero or Edge-TTS), split large files for Telegram limit (48 MB),
    log runs in logs/runs.log & logs/runs.jsonl, and send results as reply.
    """
    global ACTIVE_TASKS, QUEUE_COUNTER

    loop = asyncio.get_running_loop()
    status_msg = None
    current_stage = "QUEUED"
    voice_label = voice_conf.get("label", DEFAULT_VOICE["label"])

    tracker = RunTracker(
        user_id=user_id,
        chat_id=chat_id,
        message_id=original_message_id,
        source_filename=display_title,
        source_type=source_type,
        raw_char_count=raw_char_count,
        voice_label=voice_label
    )

    if TASK_SEMAPHORE.locked():
        QUEUE_COUNTER += 1
        pos = QUEUE_COUNTER
        status_msg = await bot.send_message(
            chat_id=chat_id,
            text=f"⏳ <b>{display_title}</b> added to queue (position: {pos}). Waiting for free worker slot...",
            parse_mode="HTML",
            reply_to_message_id=original_message_id
        )
    else:
        status_msg = await bot.send_message(
            chat_id=chat_id,
            text=f"⏳ <b>{display_title}</b>: preparing speech synthesis...",
            parse_mode="HTML",
            reply_to_message_id=original_message_id
        )

    async with TASK_SEMAPHORE:
        ACTIVE_TASKS += 1
        if QUEUE_COUNTER > 0:
            QUEUE_COUNTER = max(0, QUEUE_COUNTER - 1)

        updater = TelegramProgressUpdater(
            bot_instance=bot,
            chat_id=chat_id,
            message_id=status_msg.message_id,
            title=display_title,
            voice_label=voice_label
        )

        def sync_progress_callback(percent: int, chunk_idx: int, total_chunks: int):
            tracker.record_chunks(total_chunks)
            asyncio.run_coroutine_threadsafe(
                updater.update(percent, chunk_idx, total_chunks),
                loop
            )

        try:
            current_stage = "START_SYNTHESIS"
            await updater.update(0, 0, 1, stage="Starting synthesis")

            current_stage = "SYNTHESIS"
            t_synth_start = time.monotonic()
            output_mp3_path = await asyncio.to_thread(
                process.process_single_file,
                file_path=source_file_path,
                progress_callback=sync_progress_callback,
                voice_conf=voice_conf
            )
            t_synth_end = time.monotonic()

            current_stage = "MP3_ENCODING"
            await updater.update(100, 1, 1, stage="Encoding MP3")

            audio_duration = process.get_audio_duration(output_mp3_path)
            tracker.record_synthesis(
                synthesis_duration_sec=t_synth_end - t_synth_start,
                audio_duration_sec=audio_duration
            )

            current_stage = "SPLITTING"
            audio_parts = process.split_mp3_for_telegram(output_mp3_path)
            tracker.record_output(
                mp3_path=output_mp3_path,
                parts=audio_parts,
                get_duration_fn=process.get_audio_duration
            )

            current_stage = "TELEGRAM_UPLOAD"
            total_parts = len(audio_parts)

            for idx, part_path in enumerate(audio_parts, 1):
                part_duration = int(process.get_audio_duration(part_path))
                caption = (
                    f"✅ <b>Complete:</b> <code>{source_file_path.stem}</code>"
                    if total_parts == 1
                    else f"✅ <b>Complete (Part {idx}/{total_parts}):</b> <code>{source_file_path.stem}</code>"
                )
                title = (
                    source_file_path.stem[:40]
                    if total_parts == 1
                    else f"{source_file_path.stem[:30]} ({idx}/{total_parts})"
                )
                audio_file = FSInputFile(str(part_path))
                await bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    reply_to_message_id=original_message_id,
                    title=title,
                    performer=voice_label,
                    duration=part_duration,
                    caption=caption,
                    parse_mode="HTML"
                )

            # Clean up status message once audio delivery completes
            try:
                await bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            except Exception:
                pass

            tracker.record_success()

        except Exception as e:
            tracker.record_error(stage=current_stage, error=e)
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text=(
                        f"❌ Error processing <b>{display_title}</b>:\n"
                        f"📍 <b>Stage:</b> <code>{current_stage}</code>\n"
                        f"⚠️ <b>Details:</b> <code>{e}</code>\n\n"
                        f"<i>Run ID: {tracker.run_id}</i>"
                    ),
                    parse_mode="HTML"
                )
            except Exception:
                pass
        finally:
            ACTIVE_TASKS = max(0, ACTIVE_TASKS - 1)


async def reject_unauthorized(message: Message, user_id: int):
    """Inform unauthorized users and display their unique Telegram ID."""
    await message.answer(
        f"⛔ <b>Access restricted.</b>\n\n"
        f"Your Telegram User ID: <code>{user_id}</code>\n\n"
        f"Add this ID to your <code>.env</code> file:\n"
        f"<code>ALLOWED_USER_IDS={user_id}</code>\n"
        f"Then restart the bot.",
        parse_mode="HTML"
    )


@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Welcome message and access verification."""
    user_id = message.from_user.id
    if not is_user_allowed(user_id):
        await reject_unauthorized(message, user_id)
        return

    await message.answer(
        "👋 <b>Welcome! I am your local text-to-speech assistant.</b>\n\n"
        "Send me:\n"
        "• 📝 <b>Text message</b> or forwarded post\n"
        "• 📄 <b>.txt or .md file</b>\n"
        "• 📑 <b>.pdf document</b> (large books are automatically split into manageable parts!)\n\n"
        "I will normalize the text, synthesize high-quality speech using local neural models on your machine, "
        "and send back optimized <b>MP3 (64 kbps mono)</b> with live progress updates!\n\n"
        "<b>Diagnostic commands:</b>\n"
        "• /logs or /runs — summary of the last 5 synthesis runs with metrics and status\n"
        "• /logfile — download complete runs.log file\n\n"
        "<i>Supports parallel processing of up to 3 tasks simultaneously.</i>",
        parse_mode="HTML"
    )


@dp.message(Command("runs"))
@dp.message(Command("logs"))
async def cmd_logs(message: Message):
    """Display the last 5 synthesis runs with execution metrics and status."""
    user_id = message.from_user.id
    if not is_user_allowed(user_id):
        await reject_unauthorized(message, user_id)
        return

    if not RUNS_JSONL_PATH.exists() or RUNS_JSONL_PATH.stat().st_size == 0:
        await message.answer("ℹ️ Run history is currently empty.")
        return

    import json
    lines = []
    with open(RUNS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    lines.append(json.loads(line.strip()))
                except Exception:
                    pass

    if not lines:
        await message.answer("ℹ️ No recorded run data found.")
        return

    recent = lines[-5:]
    recent.reverse()

    blocks = ["📊 <b>RECENT TTS RUNS:</b>\n"]
    for r in recent:
        icon = "🟢" if r.get("status") == "SUCCESS" else "🔴"
        start = r.get("start_time", "")[:19].replace("T", " ")
        name = r.get("source_filename", "file")
        dur = r.get("audio_duration_sec", 0)
        dur_str = f"{int(dur)//60}m {int(dur)%60}s" if dur else "0s"
        size_mb = r.get("mp3_size_mb", 0)
        parts_info = f" (split into {r.get('parts_count')} parts)" if r.get("was_split") else ""

        block = (
            f"{icon} <code>{start}</code> | <b>{name}</b>\n"
            f"   ID: <code>{r.get('run_id')}</code>\n"
            f"   Audio: {dur_str} | Size: {size_mb} MB{parts_info}\n"
            f"   Status: <b>{r.get('status')}</b>"
        )
        if r.get("status") != "SUCCESS":
            block += f"\n   Error [{r.get('error_stage')}]: <code>{r.get('error_message')[:80]}</code>"

        blocks.append(block)

    blocks.append("\n<i>Command /logfile — download complete runs.log file.</i>")
    await message.answer("\n\n".join(blocks), parse_mode="HTML")


@dp.message(Command("logfile"))
async def cmd_logfile(message: Message):
    """Send the full runs.log file directly to the user."""
    user_id = message.from_user.id
    if not is_user_allowed(user_id):
        await reject_unauthorized(message, user_id)
        return

    if not RUNS_LOG_PATH.exists() or RUNS_LOG_PATH.stat().st_size == 0:
        await message.answer("ℹ️ File logs/runs.log is currently empty.")
        return

    doc = FSInputFile(str(RUNS_LOG_PATH), filename=f"runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    await message.answer_document(document=doc, caption="📋 Complete TTS execution log")


async def route_incoming_text(
    source_file_path: Path,
    chat_id: int,
    display_title: str,
    original_message_id: int,
    user_id: int,
    source_type: str,
    raw_char_count: int,
    extracted_text: str
):
    """
    Detect input language and route synthesis:
    - If Russian -> immediately synthesize via primary Silero model.
    - If non-Russian (UK, EN, PL, etc.) -> offer interactive language buttons with 15s timeout.
    """
    lang = detect_language(extracted_text)

    if lang == "ru":
        asyncio.create_task(
            handle_tts_job(
                source_file_path=source_file_path,
                chat_id=chat_id,
                display_title=display_title,
                original_message_id=original_message_id,
                user_id=user_id,
                source_type=source_type,
                raw_char_count=raw_char_count,
                voice_conf=DEFAULT_VOICE
            )
        )
        return

    # Non-Russian text detected: offer interactive options
    import uuid
    job_id = uuid.uuid4().hex[:8]
    lang_title = get_language_display_name(lang)

    if lang == "uk":
        voice_orig = DEFAULT_UK_VOICE
        orig_btn_text = "🗣 Synthesize in Ukrainian (Polina)"
    elif lang == "en":
        voice_orig = DEFAULT_EN_VOICE
        orig_btn_text = "🗣 Synthesize in English (Ava)"
    elif lang == "pl":
        voice_orig = DEFAULT_PL_VOICE
        orig_btn_text = "🗣 Synthesize in Polish (Zofia)"
    else:
        voice_orig = DEFAULT_EN_VOICE
        orig_btn_text = f"🗣 Synthesize in original language ({lang.upper()})"

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=orig_btn_text,
                    callback_data=f"job_choice:{job_id}:orig"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🇷🇺 Translate to Russian & synthesize (Xenia)",
                    callback_data=f"job_choice:{job_id}:trans"
                )
            ]
        ]
    )

    prompt_msg = await bot.send_message(
        chat_id=chat_id,
        text=(
            f"🌐 <b>Detected language:</b> {lang_title}\n"
            f"📄 <b>File:</b> <code>{display_title}</code> ({raw_char_count:,} chars)\n\n"
            f"How would you like to synthesize this text?\n"
            f"⏳ <i>If no choice within 15s — will auto-translate to Russian 🇷🇺</i>"
        ),
        parse_mode="HTML",
        reply_to_message_id=original_message_id,
        reply_markup=kb
    )

    PENDING_JOBS[job_id] = {
        "job_id": job_id,
        "event": asyncio.Event(),
        "choice": None,
        "source_file_path": source_file_path,
        "chat_id": chat_id,
        "display_title": display_title,
        "original_message_id": original_message_id,
        "user_id": user_id,
        "source_type": source_type,
        "raw_char_count": raw_char_count,
        "extracted_text": extracted_text,
        "prompt_message_id": prompt_msg.message_id,
        "lang": lang,
        "lang_title": lang_title,
        "voice_orig": voice_orig
    }

    asyncio.create_task(watch_job_timeout(job_id))


async def watch_job_timeout(job_id: str):
    """Wait for user selection or trigger auto-translation after 15 seconds."""
    job = PENDING_JOBS.get(job_id)
    if not job:
        return
    try:
        await asyncio.wait_for(job["event"].wait(), timeout=15.0)
    except asyncio.TimeoutError:
        if job.get("choice") is None:
            job["choice"] = "trans"
            try:
                await bot.edit_message_text(
                    chat_id=job["chat_id"],
                    message_id=job["prompt_message_id"],
                    text=(
                        f"🌐 <b>Language:</b> {job['lang_title']}\n"
                        f"📄 <b>File:</b> <code>{job['display_title']}</code>\n\n"
                        f"⏳ <i>Timed out (15s). Automatically selected translation to Russian 🇷🇺</i>"
                    ),
                    parse_mode="HTML"
                )
            except Exception:
                pass
    finally:
        choice = job.get("choice") or "trans"
        PENDING_JOBS.pop(job_id, None)
        asyncio.create_task(run_decided_job(job, choice))


@dp.callback_query(F.data.startswith("job_choice:"))
async def handle_job_choice(callback: CallbackQuery):
    """Handle interactive language choice button clicks."""
    _, job_id, choice = callback.data.split(":")
    job = PENDING_JOBS.get(job_id)
    if job and job.get("choice") is None:
        job["choice"] = choice
        job["event"].set()
        choice_label = (
            f"Synthesize in original language ({job['lang_title'].split()[0]})"
            if choice == "orig"
            else "Translate to Russian and synthesize"
        )
        await callback.answer(f"Selected: {choice_label}")
        try:
            await bot.edit_message_text(
                chat_id=job["chat_id"],
                message_id=job["prompt_message_id"],
                text=(
                    f"🌐 <b>Language:</b> {job['lang_title']}\n"
                    f"📄 <b>File:</b> <code>{job['display_title']}</code>\n\n"
                    f"✅ <b>Selection applied:</b> {choice_label}"
                ),
                parse_mode="HTML"
            )
        except Exception:
            pass
    else:
        await callback.answer("Selection already applied or timeout expired.")


async def run_decided_job(job: dict, choice: str):
    """Execute speech synthesis after user choice or timeout."""
    try:
        if "prompt_message_id" in job:
            await bot.delete_message(chat_id=job["chat_id"], message_id=job["prompt_message_id"])
    except Exception:
        pass

    if choice == "orig":
        await handle_tts_job(
            source_file_path=job["source_file_path"],
            chat_id=job["chat_id"],
            display_title=job["display_title"],
            original_message_id=job["original_message_id"],
            user_id=job["user_id"],
            source_type=job["source_type"],
            raw_char_count=job["raw_char_count"],
            voice_conf=job["voice_orig"]
        )
    else:
        trans_status_msg = await bot.send_message(
            chat_id=job["chat_id"],
            text=f"🔄 Translating <b>{job['display_title']}</b> into Russian...",
            parse_mode="HTML",
            reply_to_message_id=job["original_message_id"]
        )

        try:
            raw_text = job["extracted_text"]
            translated_text = await asyncio.to_thread(
                translate_text,
                text=raw_text,
                target_lang="ru",
                source_lang=job["lang"]
            )
        except Exception as e:
            await bot.edit_message_text(
                chat_id=job["chat_id"],
                message_id=trans_status_msg.message_id,
                text=f"❌ Translation error: {e}"
            )
            return

        clean_stem = job["source_file_path"].stem
        trans_path = INBOX_DIR / f"{clean_stem}_ru.txt"
        trans_path.write_text(translated_text, encoding="utf-8")

        # Deliver translated Russian text as markdown document attachment
        try:
            md_filename = f"{clean_stem}_translation_RU.md"
            md_path = RESULTS_DIR / md_filename
            md_path.write_text(translated_text, encoding="utf-8")
            doc = FSInputFile(str(md_path), filename=md_filename)
            await bot.send_document(
                chat_id=job["chat_id"],
                document=doc,
                caption="📝 Translated text in Russian",
                reply_to_message_id=job["original_message_id"]
            )
        except Exception as e:
            print(f"⚠️ Error sending translated .md file: {e}")

        # Archive original file to 03_Delete_it
        if job["source_file_path"].exists():
            process.move_to_delete_it(job["source_file_path"])

        try:
            await bot.delete_message(chat_id=job["chat_id"], message_id=trans_status_msg.message_id)
        except Exception:
            pass

        # Synthesize translated Russian text
        await handle_tts_job(
            source_file_path=trans_path,
            chat_id=job["chat_id"],
            display_title=trans_path.name,
            original_message_id=job["original_message_id"],
            user_id=job["user_id"],
            source_type=job["source_type"],
            raw_char_count=len(translated_text),
            voice_conf=DEFAULT_VOICE
        )


async def flush_text_buffer_after_delay(chat_id: int, delay: float):
    """Wait for typing pause to complete and combine chunks into a single document."""
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return

    buf = TEXT_BUFFERS.pop(chat_id, None)
    if not buf:
        return

    combined_text = "\n\n".join(buf["texts"]).strip()
    if not combined_text:
        return

    process.ensure_directories()
    file_stem = sanitize_filename(combined_text, max_chars=20)
    inbox_path = INBOX_DIR / f"{file_stem}.txt"

    inbox_path.write_text(combined_text, encoding="utf-8")
    display_title = f"{file_stem[:25]}.txt"

    await route_incoming_text(
        source_file_path=inbox_path,
        chat_id=buf["chat_id"],
        display_title=display_title,
        original_message_id=buf["first_message_id"],
        user_id=buf["user_id"],
        source_type="text",
        raw_char_count=len(combined_text),
        extracted_text=combined_text
    )


@dp.message(F.text | (F.caption & ~F.document))
async def handle_text_message(message: Message):
    """
    Handle plain text messages and forwarded posts (including captioned media).
    Long messages split across multiple Telegram messages (4096 char limit)
    are seamlessly merged into a cohesive document.
    """
    user_id = message.from_user.id
    if not is_user_allowed(user_id):
        await reject_unauthorized(message, user_id)
        return

    text = (message.text or message.caption or "").strip()
    if not text:
        return

    chat_id = message.chat.id
    buf = TEXT_BUFFERS.get(chat_id)

    if buf is not None:
        buf["texts"].append(text)
        if buf.get("timer_task") and not buf["timer_task"].done():
            buf["timer_task"].cancel()
    else:
        buf = {
            "chat_id": chat_id,
            "user_id": user_id,
            "first_message_id": message.message_id,
            "texts": [text],
            "timer_task": None
        }
        TEXT_BUFFERS[chat_id] = buf

    buf["timer_task"] = asyncio.create_task(flush_text_buffer_after_delay(chat_id, TEXT_BUFFER_DELAY))


@dp.message(F.document)
async def handle_document_message(message: Message):
    """Handle document uploads (.txt, .md, .pdf)."""
    user_id = message.from_user.id
    if not is_user_allowed(user_id):
        await reject_unauthorized(message, user_id)
        return

    doc: Document = message.document
    filename = doc.file_name or "document.txt"
    suffix = Path(filename).suffix.lower()

    if suffix not in [".txt", ".md", ".pdf"]:
        await message.answer(
            f"⚠️ Format <code>{suffix}</code> is not supported.\n"
            f"Please send <b>.txt</b>, <b>.md</b>, or <b>.pdf</b>.",
            parse_mode="HTML"
        )
        return

    process.ensure_directories()
    stem = Path(filename).stem
    clean_stem = re.sub(r"[^\w\dа-яА-ЯёЁa-zA-Z._-]+", "_", stem).strip("_")

    if suffix == ".pdf":
        source_type = "pdf"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf_path = Path(tmp_pdf.name)

        await bot.download(doc, destination=tmp_pdf_path)

        try:
            extracted_text = process.extract_text_from_pdf(tmp_pdf_path)
            if not extracted_text.strip():
                await message.answer("⚠️ Could not extract text from PDF (it might consist of scanned images).")
                return

            dest_path = INBOX_DIR / f"{clean_stem}.txt"
            if dest_path.exists():
                dest_path = INBOX_DIR / f"{clean_stem}_{datetime.now().strftime('%H%M%S')}.txt"

            dest_path.write_text(extracted_text, encoding="utf-8")
            display_title = dest_path.name
            raw_chars = len(extracted_text)
        finally:
            if tmp_pdf_path.exists():
                tmp_pdf_path.unlink()

    else:
        source_type = suffix.lstrip(".")
        dest_path = INBOX_DIR / f"{clean_stem}{suffix}"
        if dest_path.exists():
            dest_path = INBOX_DIR / f"{clean_stem}_{datetime.now().strftime('%H%M%S')}{suffix}"

        await bot.download(doc, destination=dest_path)
        display_title = dest_path.name
        extracted_text = dest_path.read_text(encoding="utf-8", errors="replace")
        raw_chars = len(extracted_text)

    await route_incoming_text(
        source_file_path=dest_path,
        chat_id=message.chat.id,
        display_title=display_title,
        original_message_id=message.message_id,
        user_id=message.from_user.id,
        source_type=source_type,
        raw_char_count=raw_chars,
        extracted_text=extracted_text
    )


# --- Local text ingestion endpoint for background automation ---
LOCAL_API_HOST = "127.0.0.1"
LOCAL_API_PORT = int(os.getenv("LOCAL_API_PORT", "8781"))
LOCAL_API_MAX_CHARS = 200_000
OWNER_CHAT_ID = next((int(u.strip()) for u in ALLOWED_RAW.split(",") if u.strip().isdigit()), None)
SHOW_TEXT_MAX_PARTS = 4
TELEGRAM_TEXT_LIMIT = 3800


def split_for_telegram(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list:
    """Split text into chunks so each piece fits within Telegram message size limits."""
    fits = lambda s: len(html.escape(s)) <= limit
    pieces = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if fits(para):
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?…])\s+", para):
            while not fits(sent):
                cut = limit // 2
                pieces.append(sent[:cut])
                sent = sent[cut:]
            pieces.append(sent)
    parts, cur = [], ""
    for piece in pieces:
        cand = f"{cur}\n\n{piece}" if cur else piece
        if fits(cand):
            cur = cand
        else:
            parts.append(cur)
            cur = piece
    if cur:
        parts.append(cur)
    return parts or [text[:limit]]


def quote_block(part: str) -> str:
    """Expandable blockquote representation for Telegram."""
    return f"<blockquote expandable>{html.escape(part)}</blockquote>"


async def api_speak(request: web.Request) -> web.Response:
    """POST /speak {"title": "...", "text": "..."} -> synthesizes speech and delivers MP3 to bot owner."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "JSON body required"}, status=400)
    text = str(body.get("text", "")).strip()
    title = str(body.get("title", "")).strip()[:200] or "Text"
    if not text:
        return web.json_response({"error": "Empty text"}, status=400)
    if len(text) > LOCAL_API_MAX_CHARS:
        return web.json_response({"error": f"Text exceeds maximum limit of {LOCAL_API_MAX_CHARS:,} characters"}, status=413)
    if not bot or OWNER_CHAT_ID is None:
        return web.json_response({"error": "Bot is not configured: missing BOT_TOKEN or ALLOWED_USER_IDS"}, status=503)

    process.ensure_directories()
    inbox_path = INBOX_DIR / f"{sanitize_filename(title, max_chars=40)}.txt"
    inbox_path.write_text(text, encoding="utf-8")

    head = f"🎓 <b>{html.escape(title)}</b>\n{len(text):,} chars · received via local API"
    parts = split_for_telegram(text)
    if len(parts) <= SHOW_TEXT_MAX_PARTS:
        header = await bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"{head}\n\n{quote_block(parts[0])}",
            parse_mode="HTML"
        )
        for part in parts[1:]:
            await bot.send_message(
                chat_id=OWNER_CHAT_ID,
                text=quote_block(part),
                parse_mode="HTML",
                reply_to_message_id=header.message_id
            )
    else:
        header = await bot.send_document(
            chat_id=OWNER_CHAT_ID,
            document=BufferedInputFile(text.encode("utf-8"), filename=f"{sanitize_filename(title, max_chars=40)}.txt"),
            caption=head,
            parse_mode="HTML"
        )
    await route_incoming_text(
        source_file_path=inbox_path,
        chat_id=OWNER_CHAT_ID,
        display_title=inbox_path.name,
        original_message_id=header.message_id,
        user_id=OWNER_CHAT_ID,
        source_type="hub",
        raw_char_count=len(text),
        extracted_text=text
    )
    return web.json_response({"ok": True, "chars": len(text)})


async def start_local_api() -> Optional[web.AppRunner]:
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_post("/speak", api_speak)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, LOCAL_API_HOST, LOCAL_API_PORT).start()
    except OSError as e:
        print(f"⚠️ Local API server failed to start ({LOCAL_API_HOST}:{LOCAL_API_PORT}): {e}")
        await runner.cleanup()
        return None
    print(f"Local API listening at: http://{LOCAL_API_HOST}:{LOCAL_API_PORT}/speak")
    return runner


async def main():
    """Bot entry point."""
    process.ensure_directories()
    print("=" * 60)
    print("🤖 LOCAL SPEECH SYNTHESIS TELEGRAM BOT STARTED")
    print(f"Allowed User IDs: {ALLOWED_USER_IDS if ALLOWED_USER_IDS else 'All blocked (specify in .env)'}")
    print(f"Concurrent tasks: max {MAX_CONCURRENT_TASKS}")
    print(f"Default voice: {DEFAULT_VOICE['label']}")
    print("=" * 60)

    if not bot:
        print("\n❌ ERROR: BOT_TOKEN is missing in .env file!")
        print("Open .env and set BOT_TOKEN=your_token\n")
        return

    runner = await start_local_api()
    try:
        await dp.start_polling(bot)
    finally:
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nBot stopped by user.")
