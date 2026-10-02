# m_tts: Local Neural Text-to-Speech & Telegram Bot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![aiogram](https://img.shields.io/badge/aiogram-3.x-2CA5E0.svg)](https://github.com/aiogram/aiogram)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, private, local neural **Text-to-Speech (TTS) batch processing pipeline and Telegram Bot**.

Engineered for seamless offline and low-latency speech synthesis from plain text, long-form articles, books, and lecture notes. Supports automatic language detection, translation, smart linguistic normalization (numbers, percentages, YouTube timecodes to words), Markdown cleaning, and adaptive audio splitting for Telegram upload limits.

---

## 🌟 Key Features

* **Multi-Engine Speech Synthesis**:
  * **Silero TTS (Xenia 2)**: Fast, natural, high-fidelity offline neural voice for Russian.
  * **Microsoft Edge-TTS**: Crisp cloud neural voices across multiple languages (Ukrainian, English, Polish, etc.).
  * **F5-TTS**: Advanced diffusion-based voice cloning and reference audio synthesis.
* **Intelligent Text Normalization**:
  * Strips Markdown syntax (headers, code blocks, bold/italics, quotes, URLs).
  * Automatically expands digits, percentages (e.g., `47%` → `forty-seven percent`), and timecodes (`01:23` → `one minute twenty-three seconds`) into natural spoken words.
  * Converts Roman numerals and numbered list bullets into natural speech.
  * Phonetic transcription of English brand names and abbreviations for Russian models.
  * Dual-voice mixed mode: Russian text is read by Silero while embedded English phrases are smoothly voiced by a native English model.
* **Feature-Packed Telegram Bot (`bot.py`)**:
  * Accepts plain text messages, forwarded channel posts, `.txt`, `.md`, and `.pdf` files.
  * **Live Progress Bar**: Dynamic in-place message updates with flood-wait rate limiting.
  * **Queue Management**: Processes up to 3 tasks in parallel with a FIFO waiting queue.
  * **Automatic Language Routing**: Detects language (RU, UK, EN, PL) and offers interactive inline buttons to synthesize in the original voice or translate to Russian.
  * **Auto-Split Large Audio**: Automatically slices generated MP3s larger than 48 MB without re-encoding to comply with Telegram file size limits.
  * **Security Whitelist**: Restricted access via `ALLOWED_USER_IDS` to prevent unauthorized usage.
* **Audio Optimization**:
  * Generates lightweight, speech-optimized **64 kbps mono MP3s** at 48 kHz using FFmpeg.
* **Local HTTP API (`POST /speak`)**:
  * Built-in local HTTP endpoint (`127.0.0.1:8781`) for background automation, lecture hubs, and external scripts.
* **Structured Diagnostics & Logging**:
  * Generates human-readable summaries (`logs/runs.log`) and machine-readable records (`logs/runs.jsonl`).
  * Tracks Real-Time Factor (RTF), synthesis latency, character counts, and error tracebacks.

---

## 📁 Repository Structure

```text
m_tts/
├── 01_Inbox/           # Drop input .txt/.md files here for batch synthesis
├── 02_Results/         # Generated MP3 audio files (git-ignored)
├── 03_Delete_it/       # Archived source files after processing (git-ignored)
├── logs/               # Structured execution logs: runs.log and runs.jsonl
├── models/             # Neural model weights (Silero / F5-TTS)
├── voices/             # Reference audio samples for voice cloning
├── bot.py              # Telegram bot and local HTTP ingestion API
├── process.py          # Core batch TTS engine, text normalization & CLI
├── translator.py       # Language detection and translation module
├── run_logger.py       # Structured run tracking & diagnostics
├── run.sh              # One-command batch CLI runner
├── run_bot.sh          # One-command Telegram bot runner
├── service_start.sh    # Start macOS launchd background service
├── service_status.sh   # Inspect background service status & recent logs
├── service_stop.sh     # Stop macOS launchd background service
├── .env.example        # Environment variables template
└── requirements.txt    # Python dependencies
```

---

## 🛠️ Prerequisites

* **Operating System**: macOS (Apple Silicon / Intel) or Linux.
* **Python**: `3.10` or higher (`3.12` recommended).
* **FFmpeg**: Required for audio decoding, conversion, and segment splitting.
  * macOS: `brew install ffmpeg`
  * Ubuntu/Debian: `sudo apt install ffmpeg`
* **uv** (recommended) or standard `pip`.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yevhen-prokopenko/m_tts.git
   cd m_tts
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   # Using uv:
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt

   # Or using standard python:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

   Open `.env` and fill in your details:
   ```env
   # Obtain bot token from @BotFather in Telegram
   BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ

   # Comma-separated allowed Telegram user IDs (leave empty on first run to discover your ID)
   ALLOWED_USER_IDS=348938780

   # Optional local API port (defaults to 8781)
   LOCAL_API_PORT=8781
   ```

---

## 📖 Usage Guide

### 1. Batch CLI Mode
Place `.txt` or `.md` files in `01_Inbox/` and execute:
```bash
./run.sh
# or
python process.py
```
You will be prompted to choose:
* `[1] Normal Mode`: Synthesizes speech using the primary Silero Xenia 2 voice.
* `[0] Test Run`: Synthesizes across all 8 voice and engine permutations (Silero, Edge-TTS for RU, UK, EN, PL).

Results are placed into `02_Results/`, and original source files are moved to `03_Delete_it/`.

### 2. Telegram Bot Mode
Start the Telegram bot interactively:
```bash
./run_bot.sh
# or
python bot.py
```

* Send text, articles, books, or PDFs directly to the bot.
* Use `/start` for the welcome overview.
* Use `/runs` or `/logs` to see the last 5 runs and performance metrics.
* Use `/logfile` to download the complete `runs.log`.

### 3. Background Service (macOS `launchd`)
To keep the bot running 24/7 in the background:
* **Start**: `./service_start.sh`
* **Status**: `./service_status.sh`
* **Stop**: `./service_stop.sh`

### 4. Local Automation API
Send a POST request from any script or shortcut:
```bash
curl -X POST http://127.0.0.1:8781/speak \
  -H "Content-Type: application/json" \
  -d '{"title": "Morning Briefing", "text": "Hello! Today we have three main tasks scheduled."}'
```
The text will be processed and the final audio delivered directly to your Telegram chat.

---

## 🧠 Neural Models

* **Silero TTS**: Model weights (`silero_v3_ru.pt`) are downloaded automatically on first launch if not present.
* **Edge-TTS**: Uses high-speed Microsoft neural endpoints over WebSockets (no API keys required).
* **F5-TTS (Optional)**: Checkpoints can be downloaded automatically or manually placed in `models/f5_ru/`.

---

## 🔒 Security & Privacy

* All text-to-speech processing with Silero and F5-TTS runs **100% locally** on your device.
* No private credentials, generated audio, or text archives are tracked in version control.
* Telegram bot is protected with a strict user whitelist; unauthorized users receive an access-denied message with their numeric ID.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
