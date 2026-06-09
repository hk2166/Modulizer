"""
text_cleaners.py — app-layer text normalization for XTTS synthesis.

WHY THIS EXISTS
───────────────
XTTS v2's tokenizer normalizes English well (numbers spelled out,
abbreviations expanded). For Hindi and other Indic-script languages the
tokenizer falls through to basic_cleaners (lowercase + whitespace) because
the maintainer left a TODO that was never completed. That means digits like
"25" reach the model as-is and Latin words inside Hindi text confuse the
tokenizer as it flips between Devanagari and Latin token banks mid-sentence.

This module runs BEFORE text reaches XTTS. The model still does its own
basic_cleaners pass after, but by then the text is already normalized.

WHAT WE HANDLE
──────────────
  1. Number expansion → Devanagari words       ("25" → "पच्चीस")
  2. Latin-word detection + ITRANS→Devanagari  ("coffee" → "कोफ़ी")
  3. NFC Unicode normalization for Devanagari conjuncts
  4. Whitespace collapse

WHAT WE DO NOT HANDLE (v1 scope)
─────────────────────────────────
  - Schwa-deletion (complex phonological rules; model's pretraining covers
    common cases adequately)
  - Grammatical normalization (sandhi, vibhakti)
  - English abbreviations inside Hindi ("Dr.", "km")
"""

from __future__ import annotations

import re
import unicodedata


# ── Hindi number table ────────────────────────────────────────────────────
# Hand-built because num2words 0.5.14 has no Hindi implementation.
_HI_ONES = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच",
    "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह",
    "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस",
    "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस",
    "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस",
    "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन",
    "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ",
    "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर",
    "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी",
    "छियासी", "सत्तासी", "अठासी", "नव्यासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे",
    "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

_HI_HUNDREDS = [
    "", "एक सौ", "दो सौ", "तीन सौ", "चार सौ", "पाँच सौ",
    "छह सौ", "सात सौ", "आठ सौ", "नौ सौ",
]


def _int_to_hindi(n: int) -> str:
    """
    Convert a non-negative integer to Hindi words.
    Handles 0–9,99,99,999 (crore–lakh–thousand–hundred–ones).
    Indian place-value system: crore (10^7), lakh (10^5), thousand (10^3).
    """
    if n < 0:
        return "माइनस " + _int_to_hindi(-n)
    if n <= 99:
        return _HI_ONES[n]

    parts: list[str] = []

    crore = n // 10_000_000
    n %= 10_000_000
    if crore:
        parts.append(_int_to_hindi(crore) + " करोड़")

    lakh = n // 100_000
    n %= 100_000
    if lakh:
        parts.append(_int_to_hindi(lakh) + " लाख")

    thousand = n // 1000
    n %= 1000
    if thousand:
        parts.append(_int_to_hindi(thousand) + " हज़ार")

    hundred = n // 100
    n %= 100
    if hundred:
        parts.append(_HI_HUNDREDS[hundred])

    if n > 0:
        parts.append(_HI_ONES[n])

    return " ".join(parts)


def _expand_numbers_hindi(text: str) -> str:
    """Replace stand-alone digit sequences with Hindi words."""
    def _replace(m: re.Match) -> str:
        try:
            return _int_to_hindi(int(m.group()))
        except (ValueError, OverflowError):
            return m.group()
    return re.sub(r"\b\d+\b", _replace, text)


def _has_latin(text: str) -> bool:
    """Return True if text contains any Latin-script characters."""
    return any(
        "LATIN" in unicodedata.name(ch, "")
        for ch in text if ch.isalpha()
    )


def _transliterate_latin_words(text: str) -> str:
    """
    Best-effort conversion of Latin words inside Indic-script text via
    ITRANS→Devanagari. Import is lazy — only triggered when Latin is present.
    Falls back silently if indic_transliteration is not installed.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _itrans
    except ImportError:
        return text

    def _conv(m: re.Match) -> str:
        try:
            return _itrans(m.group(0).lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
        except Exception:
            return m.group(0)

    return re.sub(r"[a-zA-Z]+", _conv, text)


def hindi_cleaners(text: str) -> str:
    """
    Normalize Hindi (Devanagari) text for XTTS synthesis.

    Pipeline: NFC normalize → expand numbers → transliterate Latin words
    → collapse whitespace.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _expand_numbers_hindi(text)
    if _has_latin(text):
        text = _transliterate_latin_words(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Language dispatch. Add Bengali, Telugu etc. here as they're implemented.
_CLEANERS: dict[str, callable] = {
    "hi": hindi_cleaners,
}


def get_cleaners(lang: str):
    """Return the cleaner function for lang, or None if no cleaner exists."""
    return _CLEANERS.get(lang)
