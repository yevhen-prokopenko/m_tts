#!/usr/bin/env python3
"""
Batch speech synthesizer supporting Silero TTS (Xenia 2 / xenia), Edge-TTS, and F5-TTS.
- Normal mode: uses the primary Silero voice — Xenia 2 (xenia)
- Test mode: automatic iteration over all supported models and voices
- Automated conversion of numbers, list numbering, and symbols to words for clear speech
- Markdown cleaning and text normalization
- Optimal speech MP3 encoding (64 kbps mono) via FFmpeg
- Source file archiving into 03_Delete_it/
"""

import os
import sys
import re
import time
import select
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np
import soundfile as sf
import num2words
from tqdm import tqdm

# Monkey-patch torchaudio.load via soundfile for stability on macOS
import torchaudio

def _soundfile_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = _soundfile_load

# Project paths
BASE_DIR = Path(__file__).resolve().parent
INBOX_DIR = BASE_DIR / "01_Inbox"
RESULTS_DIR = BASE_DIR / "02_Results"
DELETE_DIR = BASE_DIR / "03_Delete_it"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
VOICES_DIR = BASE_DIR / "voices"

SILERO_MODEL_PATH = MODELS_DIR / "silero_v3_ru.pt"
F5_DIR = MODELS_DIR / "f5_ru"
F5_CKPT_PATH = F5_DIR / "model_240000_inference.safetensors"
F5_VOCAB_PATH = F5_DIR / "vocab.txt"

# Default primary voice for normal mode (Silero Xenia)
DEFAULT_VOICE = {
    "id": "1",
    "model_type": "silero",
    "speaker_id": "xenia",
    "label": "Silero — Xenia 2 (xenia, female, RU)",
    "model_name": "Silero",
    "voice_name": "xenia",
    "lang": "ru"
}

# Default voices for Ukrainian, English, and Polish
DEFAULT_UK_VOICE = {
    "id": "3",
    "model_type": "edge_tts",
    "speaker_id": "uk-UA-PolinaNeural",
    "label": "Edge-TTS — Polina (Polina, female, UK)",
    "model_name": "EdgeTTS",
    "voice_name": "Polina",
    "lang": "uk"
}

DEFAULT_EN_VOICE = {
    "id": "5",
    "model_type": "edge_tts",
    "speaker_id": "en-US-AvaNeural",
    "label": "Edge-TTS — Ava (Ava, female, EN)",
    "model_name": "EdgeTTS",
    "voice_name": "Ava",
    "lang": "en"
}

DEFAULT_PL_VOICE = {
    "id": "7",
    "model_type": "edge_tts",
    "speaker_id": "pl-PL-ZofiaNeural",
    "label": "Edge-TTS — Zofia (Zofia, female, PL)",
    "model_name": "EdgeTTS",
    "voice_name": "Zofia",
    "lang": "pl"
}

# Catalog for test runs: 2 engines/voices each for RU, UK, EN, PL (8 options total)
TEST_CATALOG = [
    # --- Russian (2 voices) ---
    {
        "id": "1",
        "model_type": "silero",
        "speaker_id": "xenia",
        "label": "Silero — Xenia 2 (xenia, female, RU)",
        "model_name": "Silero",
        "voice_name": "xenia",
        "lang": "ru"
    },
    {
        "id": "2",
        "model_type": "edge_tts",
        "speaker_id": "ru-RU-SvetlanaNeural",
        "label": "Edge-TTS — Svetlana (Svetlana, female, RU)",
        "model_name": "EdgeTTS",
        "voice_name": "Svetlana",
        "lang": "ru"
    },
    # --- Ukrainian (2 voices) ---
    {
        "id": "3",
        "model_type": "edge_tts",
        "speaker_id": "uk-UA-PolinaNeural",
        "label": "Edge-TTS — Polina (Polina, female, UK)",
        "model_name": "EdgeTTS",
        "voice_name": "Polina",
        "lang": "uk"
    },
    {
        "id": "4",
        "model_type": "edge_tts",
        "speaker_id": "uk-UA-OstapNeural",
        "label": "Edge-TTS — Ostap (Ostap, male, UK)",
        "model_name": "EdgeTTS",
        "voice_name": "Ostap",
        "lang": "uk"
    },
    # --- English (2 voices) ---
    {
        "id": "5",
        "model_type": "edge_tts",
        "speaker_id": "en-US-AvaNeural",
        "label": "Edge-TTS — Ava (Ava, female, EN)",
        "model_name": "EdgeTTS",
        "voice_name": "Ava",
        "lang": "en"
    },
    {
        "id": "6",
        "model_type": "edge_tts",
        "speaker_id": "en-US-GuyNeural",
        "label": "Edge-TTS — Guy (Guy, male, EN)",
        "model_name": "EdgeTTS",
        "voice_name": "Guy",
        "lang": "en"
    },
    # --- Polish (2 voices) ---
    {
        "id": "7",
        "model_type": "edge_tts",
        "speaker_id": "pl-PL-ZofiaNeural",
        "label": "Edge-TTS — Zofia (Zofia, female, PL)",
        "model_name": "EdgeTTS",
        "voice_name": "Zofia",
        "lang": "pl"
    },
    {
        "id": "8",
        "model_type": "edge_tts",
        "speaker_id": "pl-PL-MarekNeural",
        "label": "Edge-TTS — Marek (Marek, male, PL)",
        "model_name": "EdgeTTS",
        "voice_name": "Marek",
        "lang": "pl"
    }
]


def ensure_directories():
    """Create required project directories if they do not exist."""
    for folder in [INBOX_DIR, RESULTS_DIR, DELETE_DIR, LOGS_DIR, MODELS_DIR, VOICES_DIR, F5_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def ensure_silero_model():
    """Ensure Silero model file is downloaded."""
    if not SILERO_MODEL_PATH.exists() or SILERO_MODEL_PATH.stat().st_size < 10_000_000:
        print("Downloading Silero TTS model into models/silero_v3_ru.pt...")
        url = "https://huggingface.co/imperialwool/silero-model-v3-ru/resolve/main/model.pt"
        subprocess.run(["curl", "-L", "--progress-bar", "-o", str(SILERO_MODEL_PATH), url], check=True)


def ensure_f5_model():
    """Ensure F5-TTS Russian model files are downloaded."""
    if not F5_VOCAB_PATH.exists() or F5_VOCAB_PATH.stat().st_size < 100:
        print("Downloading F5-TTS vocabulary vocab.txt...")
        url_vocab = "https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base/vocab.txt"
        subprocess.run(["curl", "-L", "--progress-bar", "-o", str(F5_VOCAB_PATH), url_vocab], check=True)

    if not F5_CKPT_PATH.exists() or F5_CKPT_PATH.stat().st_size < 500_000_000:
        print("Downloading F5-TTS Russian weights (1.3 GB)...")
        url_ckpt = "https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base/model_240000_inference.safetensors"
        subprocess.run(["curl", "-L", "--progress-bar", "-o", str(F5_CKPT_PATH), url_ckpt], check=True)


def ensure_f5_reference_voices(manager):
    """Generate reference audio samples for F5-TTS if not present."""
    female_wav = VOICES_DIR / "f5_female.wav"
    female_txt = VOICES_DIR / "f5_female.txt"
    male_wav = VOICES_DIR / "f5_male.wav"
    male_txt = VOICES_DIR / "f5_male.txt"

    if female_wav.exists() and female_txt.exists() and male_wav.exists() and male_txt.exists():
        return

    silero_model = manager.get_silero()
    if not female_wav.exists() or not female_txt.exists():
        text_f = "Здравствуйте! Это образец голоса для проверки нейросети и клонирования речи."
        audio_f = silero_model.apply_tts(text=text_f, speaker="baya", sample_rate=24000)
        sf.write(str(female_wav), audio_f.numpy(), 24000)
        female_txt.write_text(text_f, encoding="utf-8")

    if not male_wav.exists() or not male_txt.exists():
        text_m = "Добрый день! Это мужской образец голоса для проверки нейросети и синтеза."
        audio_m = silero_model.apply_tts(text=text_m, speaker="aidar", sample_rate=24000)
        sf.write(str(male_wav), audio_m.numpy(), 24000)
        male_txt.write_text(text_m, encoding="utf-8")


def choose_mode_and_voices(timeout: int = 10):
    """
    Interactive mode selection with 10-second timeout.
    [1] -> Normal mode (Silero Xenia 2 / xenia)
    [0] -> Test run (iterate through all models and voices)
    """
    print("\n" + "=" * 65)
    print("🎤 SPEECH SYNTHESIS MODE:")
    print(f"  [1] 🔊 Normal Mode: {DEFAULT_VOICE['label']} [DEFAULT]")
    print("  [0] 🧪 Test Run: 8 voices (RU: Silero/Edge; UK: 2x Edge; EN: 2x Edge; PL: 2x Edge)")
    print("=" * 65)

    if not sys.stdin.isatty():
        print(f"Non-interactive terminal detected. Defaulting to: {DEFAULT_VOICE['label']}\n")
        return False, [DEFAULT_VOICE]

    start_time = time.time()
    user_input = None

    while True:
        elapsed = time.time() - start_time
        remaining = int(timeout - elapsed)
        if remaining <= 0:
            break

        sys.stdout.write(f"\rSelect [1 or 0] ({remaining:2d}s remaining, Enter = [1] Normal): ")
        sys.stdout.flush()

        rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
        if rlist:
            raw = sys.stdin.readline().strip()
            if raw:
                user_input = raw
            break

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    if user_input == "0":
        print("✓ TEST RUN SELECTED: all models and voices will be synthesized!\n")
        return True, TEST_CATALOG
    else:
        print(f"✓ Normal mode selected: {DEFAULT_VOICE['label']}\n")
        return False, [DEFAULT_VOICE]


def plural_form(n: int, form1: str, form2: str, form5: str) -> str:
    """Return grammatical form for Russian number declension."""
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 10 < n_abs < 20:
        return form5
    if 1 < n1 < 5:
        return form2
    if n1 == 1:
        return form1
    return form5


def format_timecode(hours: Optional[int], minutes: int, seconds: int) -> str:
    """Format hours, minutes and seconds into natural spoken words."""
    parts = []
    if hours is not None and hours > 0:
        h_words = num2words.num2words(hours, lang="ru", gender="m")
        h_unit = plural_form(hours, "час", "часа", "часов")
        parts.append(f"{h_words} {h_unit}")

    if minutes > 0:
        m_words = num2words.num2words(minutes, lang="ru", gender="f")
        m_unit = plural_form(minutes, "минута", "минуты", "минут")
        parts.append(f"{m_words} {m_unit}")
    elif hours and hours > 0 and seconds == 0:
        pass
    elif not hours and minutes == 0 and seconds == 0:
        return "ноль секунд"

    if seconds > 0 or (not parts):
        s_words = num2words.num2words(seconds, lang="ru", gender="f")
        s_unit = plural_form(seconds, "секунда", "секунды", "секунд")
        parts.append(f"{s_words} {s_unit}")

    return " ".join(parts)


def timecodes_to_words(text: str) -> str:
    """
    Convert YouTube timecodes (HH:MM:SS and MM:SS) into spoken words.
    Example:
      01:23 -> одна минута двадцать три секунды
      1:05:30 -> один час пять минут тридцать секунд
      0:00 -> ноль секунд
    """
    # 1. HH:MM:SS (e.g. 1:23:45 or 01:05:30)
    def replace_hms(m):
        h, m_, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return format_timecode(h, m_, s)

    text = re.sub(r'\b([0-9]{1,2}):([0-5][0-9]):([0-5][0-9])\b', replace_hms, text)

    # 2. MM:SS (e.g. 01:23 or 12:05)
    def replace_ms(m):
        m_, s = int(m.group(1)), int(m.group(2))
        return format_timecode(None, m_, s)

    text = re.sub(r'\b([0-9]{1,2}):([0-5][0-9])\b', replace_ms, text)
    return text


def replace_percentages(text: str) -> str:
    """
    Convert percentages (integer and floating-point) into spoken words.
    Example:
      47% -> сорок семь процентов
      1% -> один процент
      2% -> два процента
      2.5% -> две целых пять десятых процента
    """
    # 1. Fractional percentages: 2.5% or 2,5%
    def repl_float(m):
        int_part = int(m.group(1))
        dec_part = m.group(2)
        dec_int = int(dec_part)
        dec_len = len(dec_part)

        int_word = num2words.num2words(int_part, lang="ru", gender="f")
        int_unit = plural_form(int_part, "целая", "целых", "целых")

        dec_word = num2words.num2words(dec_int, lang="ru", gender="f")
        if dec_len == 1:
            dec_unit = plural_form(dec_int, "десятая", "десятых", "десятых")
        elif dec_len == 2:
            dec_unit = plural_form(dec_int, "сотая", "сотых", "сотых")
        elif dec_len == 3:
            dec_unit = plural_form(dec_int, "тысячная", "тысячных", "тысячных")
        else:
            return f"{int_word} и {dec_word} процента"

        return f"{int_word} {int_unit} {dec_word} {dec_unit} процента"

    text = re.sub(r'\b(\d+)[.,](\d+)\s*%', repl_float, text)

    # 2. Integer percentages: 47% or 47 %
    def repl_int(m):
        val = int(m.group(1))
        val_word = num2words.num2words(val, lang="ru")
        val_unit = plural_form(val, "процент", "процента", "процентов")
        return f"{val_word} {val_unit}"

    text = re.sub(r'\b(\d+)\s*%', repl_int, text)
    return text


_CMU_DICT = None

def get_cmu_dict():
    global _CMU_DICT
    if _CMU_DICT is None:
        try:
            import cmudict
            _CMU_DICT = cmudict.dict()
        except Exception:
            _CMU_DICT = {}
    return _CMU_DICT


KNOWN_BRANDS = {
    "ai": "эй-ай",
    "api": "эй-пи-ай",
    "ui": "ю-ай",
    "ux": "ю-икс",
    "figma": "фигма",
    "youtube": "ютуб",
    "google": "гугл",
    "apple": "эппл",
    "chatgpt": "чат-джи-пи-ти",
    "gpt": "джи-пи-ти",
    "program": "програм",
    "minds": "майндз",
    "healthy": "хелси",
    "ios": "ай-о-эс",
    "android": "андроид",
    "mac": "мак",
    "windows": "виндовс",
    "web": "веб",
    "online": "онлайн",
    "offline": "офлайн",
    "pro": "про",
    "max": "макс",
    "mini": "мини",
    "pdf": "пэдээф",
}

ARPABET_TO_CYR = {
    "B": "б", "CH": "ч", "D": "д", "DH": "з", "F": "ф",
    "G": "г", "HH": "х", "JH": "дж", "K": "к", "L": "л",
    "M": "м", "N": "н", "NG": "нг", "P": "п", "R": "р",
    "S": "с", "SH": "ш", "T": "т", "TH": "с", "V": "в",
    "W": "в", "Y": "й", "Z": "з", "ZH": "ж",
    "AA": "а", "AE": "э", "AH": "а", "AO": "о", "AW": "ау",
    "AY": "ай", "EH": "е", "ER": "ер", "EY": "ей", "IH": "и",
    "IY": "и", "OW": "о", "OY": "ой", "UH": "у", "UW": "у"
}


def word_to_cyrillic(word: str) -> str:
    """Transcribe an individual English word to Cyrillic phonetic representation."""
    w_low = word.lower()
    if w_low in KNOWN_BRANDS:
        return KNOWN_BRANDS[w_low]

    cmu = get_cmu_dict()
    phones = cmu.get(w_low)
    if phones:
        res = []
        for ph in phones[0]:
            code = re.sub(r'\d+', '', ph)
            res.append(ARPABET_TO_CYR.get(code, ''))
        return ''.join(res)

    # Fallback transliteration
    simple_map = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'
    }
    return ''.join(simple_map.get(ch, ch) for ch in w_low)


def transcribe_english_to_cyrillic(text: str) -> str:
    """
    Find Latin words in text and transcribe them phonetically so the Silero
    Russian model can pronounce them naturally without skipping.
    """
    return re.sub(r'[a-zA-Z]+', lambda m: word_to_cyrillic(m.group(0)), text)


def numbers_to_words(text: str) -> str:
    """
    Convert percentages, timecodes, and digits into words.
    Required because Silero vocabulary does not contain digit characters 0-9 or % symbols.
    """
    # 0. Percentages (47% -> сорок семь процентов)
    text = replace_percentages(text)

    # 1. YouTube timecodes (HH:MM:SS and MM:SS)
    text = timecodes_to_words(text)

    # 2. Numbered lists at start of lines: 1. Text -> Один. Text
    def replace_list_num(match):
        num = int(match.group(1))
        try:
            word = num2words.num2words(num, lang="ru").capitalize()
            return f"{word}. "
        except Exception:
            return match.group(0)

    text = re.sub(r'^\s*(\d+)\.\s+', replace_list_num, text, flags=re.MULTILINE)

    # 3. Numbers inside text (e.g. 13 900 or 100 or 4)
    def replace_digits(match):
        val_str = match.group(0).replace(" ", "")
        try:
            return num2words.num2words(int(val_str), lang="ru")
        except Exception:
            return match.group(0)

    text = re.sub(r'\b\d+(?: \d{3})*\b', replace_digits, text)
    return text


def replace_underscores(text: str) -> str:
    """Speak underscore characters: Russian context in Russian, English in English."""
    # Double underscores
    text = re.sub(r'(?<=[а-яА-ЯёЁ])_{2,}', ' два подчеркивания ', text)
    text = re.sub(r'(?<=[a-zA-Z])_{2,}', ' double underscore ', text)
    text = re.sub(r'_{2,}(?=[a-zA-Z])', ' double underscore ', text)
    text = re.sub(r'_{2,}(?=[а-яА-ЯёЁ])', ' два подчеркивания ', text)
    text = re.sub(r'_{2,}', ' два подчеркивания ', text)

    # Single underscores
    text = re.sub(r'(?<=[а-яА-ЯёЁ])_', ' подчеркивание ', text)
    text = re.sub(r'(?<=[a-zA-Z])_', ' underscore ', text)
    text = re.sub(r'_(?=[a-zA-Z])', ' underscore ', text)
    text = re.sub(r'_(?=[а-яА-ЯёЁ])', ' подчеркивание ', text)
    text = re.sub(r'_', ' подчеркивание ', text)

    return re.sub(r'[ \t]{2,}', ' ', text)


def normalize_brackets_and_smileys(text: str) -> str:
    """
    Normalize brackets and smileys:
    - Balanced brackets (text) become parenthetical pauses (, text, ).
    - Smiley brackets ())), ((( are normalized to full stops.
    - Single unmatched smiley brackets become commas or periods based on context.
    """
    # 1. Balanced brackets: (explanation) -> , explanation,
    def _replace_pair(m):
        inner = m.group(1).strip()
        if not inner:
            return ""
        return f", {inner}, "

    text = re.sub(r'\(([^()]+)\)', _replace_pair, text)

    # 2. Smiley brackets 2-10 count:
    text = re.sub(r'(?<=[.!?…:;,])\s*[()]{2,10}', '', text)
    text = re.sub(r'[()]{2,10}', '. ', text)

    # 3. Single unmatched smileys:
    text = re.sub(r'(?<=[.!?…:;,])\s*[()]', '', text)
    text = re.sub(r'[()](?=\s+[а-яёa-z])', ', ', text)
    text = re.sub(r'[()]', '. ', text)

    # 4. Clean punctuation collisions
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'\.\s*,', '.', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.{2,}', '…', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def replace_common_acronyms(text: str) -> str:
    """Replace common acronyms and file extensions with spoken phonetic equivalents."""
    text = re.sub(r"(?i)\.pdf\b", " пэдээф", text)
    text = re.sub(r"(?i)\b(?:пдф|pdf)-(?=файл|документ|верси|книг)", "пэдээф-", text)
    text = re.sub(r"(?i)\b(?:пдф|pdf)-(к[аеуои]|ками|ках)\b", r"пэдээф\1", text)
    text = re.sub(r"(?i)\b(?:пдф|pdf)(к[аеуои]|ками|ках)\b", r"пэдээф\1", text)
    text = re.sub(r"(?i)\b(?:пдф|pdf)\b", "пэдээф", text)
    return text


def clean_text_for_tts(text: str, is_markdown: bool = True, lang: str = "ru") -> str:
    """Clean Markdown syntax and normalize numbers, percentages, and punctuation for speech."""
    if is_markdown:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'^[#]{1,6}\s+', '', text, flags=re.MULTILINE)
        _WORD = r"[a-zA-Z0-9а-яА-ЯёЁ]"
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(rf'(?<!{_WORD})__(?!\s)(.+?)(?<!\s)__(?!{_WORD})', r'\1', text)
        text = re.sub(rf'(?<!{_WORD})_(?!\s)(.+?)(?<!\s)_(?!{_WORD})', r'\1', text)
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<[^<]+?>', '', text)

    text = replace_underscores(text)

    if lang in ("ru", "mixed"):
        text = replace_common_acronyms(text)

    text = normalize_brackets_and_smileys(text)

    # Normalize numbers, timecodes and English transliteration for Russian Silero
    if lang == "ru":
        text = numbers_to_words(text)
        text = transcribe_english_to_cyrillic(text)
        text = re.sub(r'[\[\]]', '', text)

    return text.strip()


def split_into_chunks(text: str, max_chars: int = 350) -> list[str]:
    """
    Split text into logical sentences/chunks.
    Line breaks without punctuation are treated as sentence boundaries.
    """
    processed_lines = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        stripped = line.rstrip('"\'»”’)]} ')
        if stripped and stripped[-1] not in ".!?…:;,":
            line = line + "."
        processed_lines.append(line)

    full_text = " ".join(processed_lines)
    if not full_text:
        return []

    sentences = re.split(r'(?<=[.!?…])\s+', full_text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            subparts = re.split(r'(?<=[,;:])\s+', sent)
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                if len(part) > max_chars:
                    for word in part.split():
                        if current_len + len(word) + 1 > max_chars and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_len = len(word)
                        else:
                            current_chunk.append(word)
                            current_len += len(word) + 1
                else:
                    if current_len + len(part) + 1 > max_chars and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [part]
                        current_len = len(part)
                    else:
                        current_chunk.append(part)
                        current_len += len(part) + 1
        else:
            if current_len + len(sent) + 1 > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, sample_rate: int = 48000, bitrate: str = "64k"):
    """Convert WAV to MP3 (64 kbps mono) via FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        "-ac", "1",
        "-ar", str(sample_rate),
        str(mp3_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")


def get_audio_duration(file_path: Path) -> float:
    """Return exact audio file duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def split_mp3_for_telegram(mp3_path: Path, max_bytes: int = 48 * 1024 * 1024) -> list:
    """
    Check MP3 file size. If larger than Telegram upload limit (48 MB),
    split audio into segments without re-encoding (via ffmpeg -c copy)
    and return list of part paths.
    """
    file_size = mp3_path.stat().st_size
    if file_size <= max_bytes:
        return [mp3_path]

    import math
    target_part_size = 44 * 1024 * 1024  # 44 MB safe margin
    num_parts = max(2, math.ceil(file_size / target_part_size))
    total_duration = get_audio_duration(mp3_path)
    part_duration = (total_duration / num_parts) + 10.0

    output_dir = mp3_path.parent
    stem = mp3_path.stem

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pattern = Path(tmp_dir) / "seg_%03d.mp3"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(mp3_path),
            "-f", "segment",
            "-segment_time", str(part_duration),
            "-c", "copy",
            str(tmp_pattern)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg split error: {res.stderr}")

        seg_files = sorted(Path(tmp_dir).glob("seg_*.mp3"))
        seg_files = [s for s in seg_files if s.stat().st_size >= 50 * 1024]
        actual_total = len(seg_files)
        part_paths = []
        for idx, seg in enumerate(seg_files, 1):
            dest_name = f"{stem}_part_{idx}_of_{actual_total}.mp3"
            dest_path = output_dir / dest_name
            shutil.copy2(str(seg), str(dest_path))
            part_paths.append(dest_path)

        return part_paths


def move_to_delete_it(src_path: Path):
    """Move processed source file to 03_Delete_it/."""
    dest_path = DELETE_DIR / src_path.name
    if dest_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dest_path = DELETE_DIR / f"{src_path.stem}_{timestamp}{src_path.suffix}"
    shutil.move(str(src_path), str(dest_path))


class TTSManager:
    """Manages lifecycle and caching of Silero and F5-TTS models."""

    def __init__(self):
        self.silero_model = None
        self.f5_model = None

    def get_silero(self):
        if self.silero_model is None:
            print("⏳ Initializing Silero TTS model...")
            ensure_silero_model()
            imp = torch.package.PackageImporter(str(SILERO_MODEL_PATH))
            self.silero_model = imp.load_pickle("tts_models", "model")
            print("✓ Silero TTS loaded.")
        return self.silero_model

    def get_f5(self):
        if self.f5_model is None:
            print("⏳ Initializing F5-TTS model (Russian)...")
            ensure_f5_model()
            from f5_tts.api import F5TTS

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"  Device for F5-TTS: {device}")
            self.f5_model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=str(F5_CKPT_PATH),
                vocab_file=str(F5_VOCAB_PATH),
                device=device
            )
            print("✓ F5-TTS loaded.")
        return self.f5_model


def synthesize_audio(manager: TTSManager, voice_conf: dict, chunks: list[str], file_label: str, progress_callback=None) -> tuple[np.ndarray, int]:
    """Generate audio stream for the given list of text chunks."""
    model_type = voice_conf["model_type"]

    if model_type == "silero":
        model = manager.get_silero()
        speaker = voice_conf["speaker_id"]
        sample_rate = 48000
        silence = torch.zeros(int(sample_rate * 0.25))

        audio_parts = []
        progress = tqdm(chunks, desc=f"  {file_label[:20]} [{voice_conf['voice_name']}]", unit="chunk", ncols=80, leave=True)
        for idx, chunk in enumerate(progress):
            try:
                audio_tensor = model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate)
                audio_parts.append(audio_tensor)
                audio_parts.append(silence)
                if progress_callback:
                    pct = int((idx + 1) / len(chunks) * 100)
                    progress_callback(pct, idx + 1, len(chunks))
            except Exception as e:
                tqdm.write(f"  [Silero Error] '{chunk[:25]}...': {e}")

        if not audio_parts:
            return np.array([], dtype=np.float32), sample_rate

        full_audio = torch.cat(audio_parts).numpy()
        return full_audio, sample_rate

    elif model_type == "f5_tts":
        f5 = manager.get_f5()
        ref_file = str(BASE_DIR / voice_conf["ref_audio"])
        ref_text = (BASE_DIR / voice_conf["ref_text"]).read_text(encoding="utf-8").strip()
        sample_rate = 24000
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)

        audio_parts = []
        progress = tqdm(chunks, desc=f"  {file_label[:20]} [{voice_conf['voice_name']}]", unit="chunk", ncols=80, leave=True)
        for idx, chunk in enumerate(progress):
            try:
                wav, sr, _ = f5.infer(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    gen_text=chunk,
                    nfe_step=16,
                    seed=42,
                    show_info=lambda *args, **kwargs: None
                )
                audio_parts.append(wav)
                audio_parts.append(silence)
                if progress_callback:
                    pct = int((idx + 1) / len(chunks) * 100)
                    progress_callback(pct, idx + 1, len(chunks))
            except Exception as e:
                tqdm.write(f"  [F5-TTS Error] '{chunk[:25]}...': {e}")

        if not audio_parts:
            return np.array([], dtype=np.float32), sample_rate

        full_audio = np.concatenate(audio_parts)
        return full_audio, sample_rate

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# --- Mixed text: Russian voice for Cyrillic, English voice for Latin phrases ---
INLINE_EN_SPEAKER = os.getenv("INLINE_EN_SPEAKER", "en-US-AvaNeural")
INLINE_EN_MIN_WORDS = 1
_EN_WORD = r"[A-Za-z][A-Za-z0-9'’\-]*"
EN_PHRASE_RE = re.compile(rf"{_EN_WORD}(?:[ \t,:;\"“”()\-–—]+{_EN_WORD})*[.!?]?")
SEGMENT_PAUSE_SEC = 0.015


def trim_silence(audio: np.ndarray, threshold: float = 0.005, pad_ms: int = 15, sample_rate: int = 48000) -> np.ndarray:
    """Trim silence at beginning and end of audio chunk, preserving micro-fade padding."""
    if not audio.size:
        return audio
    idx = np.where(np.abs(audio) > threshold)[0]
    if not idx.size:
        return audio
    pad = int(sample_rate * (pad_ms / 1000))
    start = max(0, idx[0] - pad)
    end = min(len(audio), idx[-1] + pad)
    return audio[start:end].copy()


def split_mixed_segments(chunk: str) -> list[tuple[str, str]]:
    """Split chunk into [("ru", text), ("en", phrase), ...]."""
    segments, pos = [], 0
    for m in EN_PHRASE_RE.finditer(chunk):
        if len(re.findall(_EN_WORD, m.group(0))) < INLINE_EN_MIN_WORDS:
            continue
        if m.start() > pos:
            segments.append(("ru", chunk[pos:m.start()]))
        segments.append(("en", m.group(0).strip()))
        pos = m.end()
    if pos < len(chunk):
        segments.append(("ru", chunk[pos:]))
    return segments


def normalize_ru_segment(text: str) -> str:
    text = numbers_to_words(text)
    text = transcribe_english_to_cyrillic(text)
    return re.sub(r'[\[\]]', '', text).strip()


def decode_to_float(mp3_path: Path, sample_rate: int) -> np.ndarray:
    """Decode MP3 to float32 mono array at target sample rate via FFmpeg."""
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp3_path), "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True, check=True)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def synthesize_en_phrases(phrases: list[str], sample_rate: int) -> dict:
    """Synthesize unique English phrases via Edge-TTS with silence trimming."""
    import edge_tts
    import asyncio

    out = {}

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, phrase in enumerate(phrases):
                path = Path(tmp_dir) / f"en_{idx:04d}.mp3"
                try:
                    await edge_tts.Communicate(phrase, INLINE_EN_SPEAKER).save(str(path))
                    raw_audio = decode_to_float(path, sample_rate)
                    out[phrase] = trim_silence(raw_audio, sample_rate=sample_rate)
                except Exception as e:
                    tqdm.write(f"  [Edge-TTS EN] '{phrase[:30]}': {e} — falling back to transliteration")

    if phrases:
        asyncio.run(_run())
    return out


def match_loudness(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Match loudness of English insert to Russian voice (RMS based, preventing clipping)."""
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms <= 1e-6 or target_rms <= 1e-6:
        return audio
    gain = target_rms / rms
    peak = float(np.max(np.abs(audio))) * gain
    if peak > 0.98:
        gain *= 0.98 / peak
    return (audio * gain).astype(np.float32)


def synthesize_mixed_silero(manager: TTSManager, voice_conf: dict, chunks: list[str], file_label: str, progress_callback=None) -> tuple[np.ndarray, int]:
    """Synthesize Silero Russian voice with inline native English audio for Latin phrases."""
    model = manager.get_silero()
    speaker = voice_conf["speaker_id"]
    sample_rate = 48000
    chunk_pause = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    seg_pause = np.zeros(int(sample_rate * SEGMENT_PAUSE_SEC), dtype=np.float32)

    plan = [split_mixed_segments(c) for c in chunks]
    phrases = list(dict.fromkeys(t for segs in plan for lang, t in segs if lang == "en"))
    en_audio = synthesize_en_phrases(phrases, sample_rate)

    parts, ru_rms = [], []
    pending_en = []
    progress = tqdm(plan, desc=f"  {file_label[:20]} [{voice_conf['voice_name']}+EN]", unit="chunk", ncols=80, leave=True)
    for idx, segs in enumerate(progress):
        for lang, text in segs:
            if lang == "en" and text in en_audio:
                pending_en.append(len(parts))
                parts.extend([en_audio[text], seg_pause])
                continue
            ru_text = normalize_ru_segment(text)
            if not re.search(r"[а-яА-ЯёЁa-zA-Z]", ru_text):
                continue
            try:
                audio = model.apply_tts(text=ru_text, speaker=speaker, sample_rate=sample_rate).numpy().astype(np.float32)
                trimmed_ru = trim_silence(audio, sample_rate=sample_rate)
                ru_rms.append(float(np.sqrt(np.mean(trimmed_ru ** 2))))
                parts.extend([trimmed_ru, seg_pause])
            except Exception as e:
                tqdm.write(f"  [Silero Error] '{ru_text[:25]}...': {e}")
        parts.append(chunk_pause)
        if progress_callback:
            progress_callback(int((idx + 1) / len(plan) * 100), idx + 1, len(plan))

    if ru_rms:
        target = float(np.median(ru_rms))
        for i in pending_en:
            parts[i] = match_loudness(parts[i], target)
    if not any(p.size for p in parts):
        return np.array([], dtype=np.float32), sample_rate
    return np.concatenate(parts), sample_rate


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t.strip())
    return "\n\n".join(pages)


def synthesize_edge_tts_to_mp3(
    chunks: list,
    speaker: str,
    output_mp3: Path,
    file_label: str = "",
    progress_callback=None
):
    """Synthesize text chunks via Edge-TTS and assemble into a 64 kbps mono MP3."""
    import edge_tts
    import asyncio

    async def _async_synth():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_files = []
            progress = tqdm(chunks, desc=f"  {file_label[:20]} [{speaker}]", unit="chunk", ncols=80, leave=True)
            for idx, chunk in enumerate(progress):
                part_path = Path(tmp_dir) / f"chunk_{idx:04d}.mp3"
                comm = edge_tts.Communicate(chunk, speaker)
                await comm.save(str(part_path))
                tmp_files.append(part_path)
                if progress_callback:
                    pct = int((idx + 1) / len(chunks) * 100)
                    progress_callback(pct, idx + 1, len(chunks))

            concat_txt = Path(tmp_dir) / "concat.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for p in tmp_files:
                    f.write(f"file '{p.name}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt),
                "-codec:a", "libmp3lame",
                "-b:a", "64k",
                "-ac", "1",
                "-ar", "48000",
                str(output_mp3)
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg error concatenating Edge-TTS: {res.stderr}")

    asyncio.run(_async_synth())


def process_single_file(
    file_path: Path,
    progress_callback=None,
    voice_conf: dict = DEFAULT_VOICE,
    manager: TTSManager = None
) -> Path:
    """
    Synthesize an individual text or markdown file, invoking progress_callback,
    saving MP3 in 02_Results/, archiving source to 03_Delete_it/, and returning MP3 path.
    """
    ensure_directories()
    if manager is None:
        manager = TTSManager()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp1251", errors="replace") as f:
            content = f.read()

    is_markdown = file_path.suffix.lower() == ".md"
    model_type = voice_conf.get("model_type", "silero")
    mixed = model_type == "silero" and voice_conf.get("lang", "ru") == "ru"
    cleaned_text = clean_text_for_tts(content, is_markdown=is_markdown, lang="mixed" if mixed else voice_conf.get("lang", "ru"))
    chunks = split_into_chunks(cleaned_text, max_chars=350)

    if not chunks:
        move_to_delete_it(file_path)
        raise ValueError("File contains no text for speech synthesis")

    out_filename = f"{file_path.stem}.mp3"
    output_mp3 = RESULTS_DIR / out_filename

    if model_type == "edge_tts":
        synthesize_edge_tts_to_mp3(
            chunks=chunks,
            speaker=voice_conf["speaker_id"],
            output_mp3=output_mp3,
            file_label=file_path.stem,
            progress_callback=progress_callback
        )
    else:
        synth = synthesize_mixed_silero if mixed else synthesize_audio
        audio_data, sr = synth(
            manager, voice_conf, chunks,
            file_label=file_path.stem,
            progress_callback=progress_callback
        )
        if audio_data.size == 0:
            raise RuntimeError("Audio generation error")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        try:
            sf.write(str(tmp_wav_path), audio_data, sr)
            convert_wav_to_mp3(tmp_wav_path, output_mp3, sample_rate=48000, bitrate="64k")
        finally:
            if tmp_wav_path.exists():
                tmp_wav_path.unlink()

    move_to_delete_it(file_path)
    return output_mp3


def process_files():
    """Main batch processing entry point for inbox files."""
    ensure_directories()
    manager = TTSManager()

    inbox_files = sorted([
        f for f in INBOX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".txt", ".md"]
    ])

    if not inbox_files:
        print("\n📂 No files found for processing (.txt or .md) in 01_Inbox/ directory.")
        print("Place text files in '01_Inbox' and run the script again.\n")
        return

    print(f"\n📁 Discovered files for speech synthesis: {len(inbox_files)}")
    for idx, f in enumerate(inbox_files, 1):
        print(f"  {idx}. {f.name} ({f.stat().st_size} bytes)")

    # Interactive mode selection (normal or test catalog)
    is_test_mode, selected_voices = choose_mode_and_voices(timeout=10)

    # Ensure reference audio samples exist if F5-TTS is included
    if any(v["model_type"] == "f5_tts" for v in selected_voices):
        ensure_f5_reference_voices(manager)

    # Process each discovered file
    for file_idx, file_path in enumerate(inbox_files, 1):
        print(f"\n[{file_idx}/{len(inbox_files)}] File: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp1251", errors="replace") as f:
                content = f.read()

        is_markdown = file_path.suffix.lower() == ".md"
        cleaned_text = clean_text_for_tts(content, is_markdown=is_markdown)
        chunks = split_into_chunks(cleaned_text, max_chars=350)

        if not chunks:
            print("  ⚠️ File is empty or contains no pronounceable text. Skipping.\n")
            move_to_delete_it(file_path)
            continue

        print(f"  Chunks: {len(chunks)}. Voice options to render ({len(selected_voices)}):")

        for v_idx, voice_conf in enumerate(selected_voices, 1):
            v_desc = f"[{v_idx}/{len(selected_voices)}] {voice_conf['model_name']} - {voice_conf['voice_name']} ({voice_conf.get('lang', 'ru')})"
            print(f"\n  👉 {v_desc}:")

            if is_test_mode:
                out_filename = f"{voice_conf['model_name']}_{voice_conf['voice_name']}_{file_path.stem}.mp3"
            else:
                out_filename = f"{file_path.stem}.mp3"

            output_mp3 = RESULTS_DIR / out_filename

            v_lang = voice_conf.get("lang", "ru")
            voice_chunks = chunks
            if is_test_mode:
                from translator import detect_language, translate_text
                source_lang = detect_language(content)
                if v_lang != source_lang:
                    print(f"  🌐 Translating chunks into {v_lang.upper()} for test voice...")
                    translated_content = translate_text(content, target_lang=v_lang, source_lang=source_lang)
                    cleaned_tr = clean_text_for_tts(translated_content, is_markdown=is_markdown, lang=v_lang)
                    voice_chunks = split_into_chunks(cleaned_tr, max_chars=350)

            if voice_conf.get("model_type") == "edge_tts":
                synthesize_edge_tts_to_mp3(
                    chunks=voice_chunks,
                    speaker=voice_conf["speaker_id"],
                    output_mp3=output_mp3,
                    file_label=file_path.stem
                )
                print(f"  ✓ MP3 (64 kbps) saved: 02_Results/{output_mp3.name}")
            else:
                audio_data, sr = synthesize_audio(manager, voice_conf, voice_chunks, file_label=file_path.stem)
                if audio_data.size == 0:
                    print("  ⚠️ Skipping voice variant due to synthesis error.")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    tmp_wav_path = Path(tmp_wav.name)

                try:
                    sf.write(str(tmp_wav_path), audio_data, sr)
                    convert_wav_to_mp3(tmp_wav_path, output_mp3, sample_rate=48000, bitrate="64k")
                    print(f"  ✓ MP3 (64 kbps) saved: 02_Results/{output_mp3.name}")
                finally:
                    if tmp_wav_path.exists():
                        tmp_wav_path.unlink()

        # Move source file to 03_Delete_it only after all variants are processed
        move_to_delete_it(file_path)
        print(f"\n  ✓ Source moved: 03_Delete_it/{file_path.name}")

    print("\n" + "=" * 65)
    print("🎉 All tasks completed successfully!")
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    process_files()
