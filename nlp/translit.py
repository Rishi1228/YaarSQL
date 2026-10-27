"""Devanagari -> Roman (Hinglish-style) transliteration.

Used in Phase 3 to turn the Hindi POS corpus into *Romanized* Hindi, giving the Hinglish
tagger far more Hindi training words than our small hand-tagged set.

Rules: each consonant carries an inherent "a" unless a vowel sign (matra) or virama follows.
Hindi schwa deletion: the final inherent "a" is dropped (कमल -> kamal), and a medial inherent "a"
is dropped between two vowelled syllables (कितने -> kitne, not kitane).
"""
import unicodedata

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y",
}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
          "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o"}
MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai",
          "ो": "o", "ौ": "au", "ॉ": "o"}
VIRAMA, NUKTA = "्", "़"
NASAL = {"ं": "n", "ँ": "n", "ः": "h"}
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# Common words whose usual Hinglish spelling the rules don't produce
CONVENTIONAL = {"में": "mein", "हैं": "hain", "है": "hai", "नहीं": "nahi", "यह": "yeh", "वह": "woh",
                "और": "aur", "भी": "bhi", "ही": "hi", "को": "ko", "के": "ke", "की": "ki", "का": "ka",
                "से": "se", "ने": "ne", "पर": "par", "कि": "ki", "कंप्यूटर": "computer"}


def transliterate_word(word):
    word = unicodedata.normalize("NFC", word).translate(DIGITS)
    if word in CONVENTIONAL:
        return CONVENTIONAL[word]
    # units: [consonant_roman, vowel_roman, inherent?]  or ["", vowel, False] for independent vowels
    units = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if nxt == NUKTA and ch + NUKTA in CONS:
            ch, i = ch + NUKTA, i + 1
            nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONS:
            if nxt == VIRAMA:
                units.append([CONS[ch], "", False])
                i += 2
                continue
            if nxt in MATRAS:
                units.append([CONS[ch], MATRAS[nxt], False])
                i += 2
                continue
            units.append([CONS[ch], "a", True])
        elif ch in VOWELS:
            units.append(["", VOWELS[ch], False])
        elif ch in NASAL and units:
            units[-1][1] += NASAL[ch]
            units[-1][2] = False
        elif ch in MATRAS or ch == VIRAMA:
            pass
        else:
            units.append([ch, "", False])
        i += 1

    # schwa deletion (kept after a consonant cluster: छात्र -> chhatra, मंत्र -> mantra)
    if len(units) > 1 and units[-1][2] and units[-2][1]:
        units[-1][1], units[-1][2] = "", False
    for j in range(len(units) - 2, 0, -1):
        if units[j][2] and units[j - 1][1] and units[j + 1][1] and units[j + 1][0]:
            units[j][1], units[j][2] = "", False
    return "".join(c + v for c, v, _ in units)


def transliterate(text):
    return " ".join(transliterate_word(w) for w in text.split())
