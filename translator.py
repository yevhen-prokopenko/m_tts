#!/usr/bin/env python3
"""
Language detection and neural machine translation module.
- Fast and reliable language detection: Russian, Ukrainian, Polish, English, and more.
- High-quality translation for long texts into Russian (or any target language)
  with automated paragraph and sentence splitting.
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional, List
from langdetect import detect
from deep_translator import MyMemoryTranslator

LANGUAGE_NAMES = {
    "ru": "Russian 🇷🇺",
    "uk": "Ukrainian 🇺🇦",
    "ua": "Ukrainian 🇺🇦",
    "en": "English 🇬🇧",
    "de": "German 🇩🇪",
    "fr": "French 🇫🇷",
    "es": "Spanish 🇪🇸",
    "it": "Italian 🇮🇹",
    "pl": "Polish 🇵🇱"
}


def get_language_display_name(lang_code: str) -> str:
    """Return a human-readable language name with country flag."""
    code = lang_code.lower()
    return LANGUAGE_NAMES.get(code, f"Foreign ({code.upper()}) 🌐")


UK_DISTINCT_WORDS = {
    "перший", "другий", "третій", "четвертий", "п’ятий", "пятий", "шостий",
    "сьомий", "восьмий", "дев’ятий", "девятий", "десятий",
    "це", "якщо", "але", "про", "від", "для", "також", "вже", "чи",
    "його", "їх", "вони", "ми", "ви", "ти", "було", "буде", "був", "була",
    "цього", "тому", "після", "коли", "тут", "зараз", "дуже", "багато",
    "чому", "хто", "що", "який", "яка", "яке", "які", "лише", "тільки"
}

RU_DISTINCT_WORDS = {
    "первый", "второй", "третий", "четвертый", "пятый", "шестой", "седьмой",
    "восьмой", "девятый", "десятый", "это", "если", "но", "про", "от", "для",
    "также", "уже", "или", "его", "их", "они", "было", "будет", "был", "была",
    "этого", "потому", "после", "когда", "здесь", "сейчас", "очень", "много",
    "почему", "кто", "что", "какой", "какая", "какое", "какие", "только"
}

PL_DISTINCT_WORDS = {
    "się", "nie", "jest", "bardzo", "tylko", "przez", "jeszcze", "kiedy",
    "tutaj", "teraz", "dlaczego", "gdzie", "jaki", "jaka", "jakie",
    "który", "która", "które", "proszę", "dziękuję", "albo", "oraz",
    "będzie", "było", "był", "była", "jego", "ich", "oni", "dla", "może"
}


def detect_language(text: str) -> str:
    """
    Detects language with heuristics for Russian, Ukrainian, Polish, and English.
    Returns two-letter ISO language code: 'ru', 'uk', 'pl', 'en', etc.
    """
    cleaned = text.strip()
    if not cleaned:
        return "ru"

    # 1. Precise heuristics based on distinct Slavic alphabet characters
    uk_chars = len(re.findall(r"[іїєґІЇЄҐ]", cleaned))
    ru_chars = len(re.findall(r"[ыэъёЫЭЪЁ]", cleaned))

    if uk_chars > 0 and uk_chars >= ru_chars:
        return "uk"
    if ru_chars > 0 and ru_chars > uk_chars:
        return "ru"

    # Polish unique diacritics (ąćęłńóśźż)
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁіІїЇєЄґҐ]", cleaned))
    pl_chars = len(re.findall(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", cleaned))
    if pl_chars > 0 and cyrillic_count == 0:
        return "pl"

    # 2. Check against frequency dictionary word sets
    words_cyr = set(re.findall(r"[а-яА-ЯёЁіІїЇєЄґҐ]+", cleaned.lower()))
    uk_matches = len(words_cyr.intersection(UK_DISTINCT_WORDS))
    ru_matches = len(words_cyr.intersection(RU_DISTINCT_WORDS))

    if uk_matches > ru_matches:
        return "uk"
    if ru_matches > uk_matches:
        return "ru"

    words_lat = set(re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", cleaned.lower()))
    if len(words_lat.intersection(PL_DISTINCT_WORDS)) >= 2 and cyrillic_count == 0:
        return "pl"

    # 3. Detection via langdetect library (first 2000 chars)
    try:
        sample = cleaned[:2000]
        detected = detect(sample)
        if detected in ["uk", "ua"]:
            return "uk"
        if detected == "ru":
            return "ru"
        if detected in ["en", "de", "fr", "es", "it", "pl"]:
            return detected
    except Exception:
        pass

    # 4. Fallback ratio: Latin vs Cyrillic
    latin_count = len(re.findall(r"[a-zA-Z]", cleaned))

    if latin_count > cyrillic_count:
        return "en"
    return "ru"


def _translate_single_chunk(text: str, target_lang: str = "ru", source_lang: str = "auto") -> str:
    """Translates a single chunk of text (up to 3000 characters)."""
    src = "auto" if source_lang in ["auto", "", None] else source_lang
    if src == "ua":
        src = "uk"

    # Primary method: fast Google Translate single API
    try:
        encoded_q = urllib.parse.quote(text)
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={src}&tl={target_lang}&dt=t&q={encoded_q}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result_parts = [part[0] for part in data[0] if part and part[0]]
            return "".join(result_parts)
    except Exception:
        pass

    # Fallback method: MyMemoryTranslator
    try:
        mm_src = "en" if src == "auto" else src
        mm = MyMemoryTranslator(source=mm_src, target=target_lang)
        return mm.translate(text)
    except Exception as e:
        print(f"[Translator Warning] Translation error: {e}")
        return text


def split_text_for_translation(text: str, max_chars: int = 2500) -> List[str]:
    """Splits long text by paragraphs and sentences for translation."""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            continue

        if len(para_clean) > max_chars:
            # Split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?…])\s+", para_clean)
            for s in sentences:
                if current_len + len(s) + 1 > max_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [s]
                    current_len = len(s)
                else:
                    current_chunk.append(s)
                    current_len += len(s) + 1
        else:
            if current_len + len(para_clean) + 1 > max_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [para_clean]
                current_len = len(para_clean)
            else:
                current_chunk.append(para_clean)
                current_len += len(para_clean) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


def translate_text(
    text: str,
    target_lang: str = "ru",
    source_lang: str = "auto",
    progress_callback=None
) -> str:
    """
    Translates long text of arbitrary length into target language (default Russian),
    chunking into semantic blocks and joining the output.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    chunks = split_text_for_translation(cleaned, max_chars=2500)
    if not chunks:
        return cleaned

    translated_parts = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        tr = _translate_single_chunk(chunk, target_lang=target_lang, source_lang=source_lang)
        translated_parts.append(tr)
        if progress_callback:
            pct = int((idx / total) * 100)
            progress_callback(pct, idx, total)

    return "\n\n".join(translated_parts)
