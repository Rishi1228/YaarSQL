"""Phase 1 — Text preprocessing for English, Hindi (Devanagari) and Hinglish (Romanized Hindi-English).

Pipeline (each step is a separate function so the website can show them one by one):
    1. normalize_unicode   NFC, Devanagari digits -> 0-9, nukta / chandrabindu folding, quotes
    2. lowercase           Latin script only (Devanagari has no case)
    3. split_sentences     handles . ? ! and the Hindi danda (।)
    4. tokenize            regex tokenizer aware of numbers, Devanagari words and quoted values
    5. tag_scripts         LATIN / DEVANAGARI / NUM / PUNCT per token
    6. detect_language     sentence-level: en / hi / hg (rule-based; Phase 2 adds an n-gram model)
    7. tag_word_language   token-level: EN / HI / HI-ROM / NUM / PUNCT / UNK
    8. normalize_hinglish  maps spelling variants (kitney, ktne -> kitne) using a lexicon + edit distance
    9. numbers_to_digits   five / paanch / पाँच -> 5
   10. remove_stopwords    optional (the n-gram and HMM phases keep function words)
   11. stem                Porter (English) + suffix stripping (Hindi and Hinglish)
"""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

LEX_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"

# ---------------------------------------------------------------------------
# Lexicons (loaded once)
# ---------------------------------------------------------------------------


def _read_list(name):
    lines = (LEX_DIR / name).read_text(encoding="utf-8").splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


@lru_cache(maxsize=1)
def lexicons():
    variants = json.loads((LEX_DIR / "hinglish_variants.json").read_text(encoding="utf-8"))
    variants.pop("_about", None)
    variant_to_canon = {}
    for canon, forms in variants.items():
        for f in forms:
            if " " not in f:  # single-token variants only
                variant_to_canon.setdefault(f, canon)
    schema = json.loads((LEX_DIR / "schema.json").read_text(encoding="utf-8"))
    english = _read_list("english_vocab.txt")
    # canonical keys like "kitne" are Hindi, but some (students, average) are English words
    hinglish = _read_list("hinglish_words.txt") | {c for c in variants if c not in english}
    # every Romanized Hindi word in the hand-tagged Hinglish corpus counts as known Hinglish
    pos_file = LEX_DIR.parent / "pos" / "hinglish_gold.txt"
    if pos_file.exists():
        for line in pos_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                hinglish |= {tok.rpartition("/")[0].lower() for tok in line.split()
                             if tok.rpartition("/")[0].isalpha() and tok.rpartition("/")[0].lower() not in english}
    return {
        "variant_to_canon": variant_to_canon,
        "canon": set(variants),
        "hinglish": hinglish,
        "english": english,
        # words valid in both languages (to, the, do, par) or variants that are English words (me, he)
        "ambiguous": (hinglish & english) | {v for v in variant_to_canon if v in english},
        "number_words": schema["number_words"],
    }


# ---------------------------------------------------------------------------
# 1-2. Normalization
# ---------------------------------------------------------------------------
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUKTA = "़"            # ़  (ज़ -> ज, फ़ -> फ)
CHANDRABINDU = "ँ"     # ँ  folded to anusvara ं (पाँच -> पांच)
ANUSVARA = "ं"
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_unicode(text, fold_nukta=True):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(DEVANAGARI_DIGITS).translate(QUOTES)
    if fold_nukta:
        text = unicodedata.normalize("NFD", text).replace(NUKTA, "")
        text = unicodedata.normalize("NFC", text)
        text = text.replace(CHANDRABINDU, ANUSVARA)
    text = text.replace("‌", "").replace("‍", "")  # zero-width joiners
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text):
    # str.lower() leaves Devanagari untouched, so this is safe for mixed text
    return text.lower()


# ---------------------------------------------------------------------------
# 3-4. Sentence splitting and tokenization
# ---------------------------------------------------------------------------
SENT_END = re.compile(r"(?<=[.?!।॥])\s+")

DEVA = r"ऀ-ॣ०-ॿ"     # Devanagari block minus the danda marks
TOKEN_RE = re.compile(
    rf"""
    (?P<QUOTED>'[^']+'|"[^"]+")             # quoted values: 'Computer'
  | (?P<ORD>\d+(?:st|nd|rd|th)\b)          # ordinals: 1st, 2nd
  | (?P<NUM>\d+(?:[.,]\d+)*)                 # 8.5, 1,20,000
  | (?P<DEVANAGARI>[{DEVA}]+)                # Hindi words (matras stay attached)
  | (?P<LATIN>[a-zA-Z]+(?:'[a-zA-Z]+)?)     # English / Hinglish words, don't
  | (?P<PUNCT>[^\s\w]|[।॥])                  # any other single symbol
    """,
    re.VERBOSE,
)


def split_sentences(text):
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def tokenize(text):
    """Return a list of (token, script) pairs."""
    out = []
    for m in TOKEN_RE.finditer(text):
        kind, tok = m.lastgroup, m.group()
        if kind == "QUOTED":
            tok, kind = tok[1:-1], "LATIN" if re.search(r"[a-zA-Z]", tok) else "DEVANAGARI"
        elif kind == "ORD":
            kind = "LATIN"
        elif kind == "NUM":
            tok = tok.replace(",", "")
        out.append((tok, kind))
    return out


def tag_scripts(tokens):
    return [script for _, script in tokens]


# ---------------------------------------------------------------------------
# 6-7. Language identification (rule-based; replaced by n-gram LID in Phase 2)
# ---------------------------------------------------------------------------


def detect_language(tokens):
    """Sentence-level language: 'hi' (Devanagari), 'hg' (Hinglish) or 'en'."""
    lex = lexicons()
    words = [(t.lower(), s) for t, s in tokens if s in ("LATIN", "DEVANAGARI")]
    if not words:
        return "en"
    deva = sum(s == "DEVANAGARI" for _, s in words)
    if deva / len(words) >= 0.5:
        return "hi"
    latin = [w for w, s in words if s == "LATIN"]
    def is_hindi(w):
        if w in lex["ambiguous"]:
            return False
        canon = lex["variant_to_canon"].get(w, w)
        return canon in lex["hinglish"]
    hindi_like = sum(1 for w in latin if is_hindi(w))
    return "hg" if hindi_like >= 1 else "en"


def tag_word_language(tokens, sent_lang):
    """Token-level language tags, used to show code-mixing."""
    lex = lexicons()
    tags = []
    for tok, script in tokens:
        w = tok.lower()
        if script == "NUM":
            tags.append("NUM")
        elif script == "PUNCT":
            tags.append("PUNCT")
        elif script == "DEVANAGARI":
            tags.append("HI")
        elif lex["variant_to_canon"].get(w, "") in lex["english"]:
            tags.append("EN")    # variant of an English word: dept -> department
        elif w in lex["ambiguous"]:
            tags.append("HI-ROM" if sent_lang == "hg" else "EN")
        elif w in lex["variant_to_canon"]:
            tags.append("HI-ROM")
        elif w in lex["hinglish"]:
            tags.append("HI-ROM")
        elif w in lex["english"]:
            tags.append("EN")
        else:
            tags.append("UNK")   # names, typos, rare words
    return tags


# ---------------------------------------------------------------------------
# 8. Hinglish spelling normalization
# ---------------------------------------------------------------------------


def edit_distance(a, b):
    """Levenshtein distance (dynamic programming, O(len(a)*len(b)))."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_hinglish(words):
    """Map Hinglish spelling variants to one canonical form.

    1. exact lexicon lookup (kitney -> kitne)
    2. otherwise, for unknown words of length >= 5, the closest canonical word at edit distance 1
    Returns (normalized_words, list of (old, new, rule) changes).
    """
    lex = lexicons()
    out, changes = [], []
    for w in words:
        lw = w.lower()
        if lw in lex["variant_to_canon"]:
            new = lex["variant_to_canon"][lw]
            out.append(new)
            if new != lw:
                changes.append((w, new, "lexicon"))
            continue
        known = lw in lex["hinglish"] or lw in lex["english"] or not lw.isalpha()
        if not known and len(lw) >= 5:
            best = min(lex["canon"], key=lambda c: edit_distance(lw, c))
            if edit_distance(lw, best) == 1:
                out.append(best)
                changes.append((w, best, "edit-distance"))
                continue
        out.append(w)
    return out, changes


# ---------------------------------------------------------------------------
# 9. Number words -> digits
# ---------------------------------------------------------------------------


def numbers_to_digits(words, lang):
    table = lexicons()["number_words"].get(lang, {})
    if lang == "hi":
        # number words were stored before nukta/chandrabindu folding; fold them the same way
        table = {normalize_unicode(k): v for k, v in table.items()}
    return [str(table[w]) if w in table else w for w in words]


# ---------------------------------------------------------------------------
# 10. Stopwords
# ---------------------------------------------------------------------------
STOPWORDS = {
    "en": set("""a an the is are was were be been being am do does did of in on at to for from by with
        about as into than then so and or but if this that these those there here it its me my i you
        your we our they them their he she his her him all any some such only own same very can will
        just should now please tell give show""".split()),
    "hi": set("""का की के को से में पर तक और या भी ही तो है हैं था थी थे हो होगा यह वह ये वे इस उस
        इन उन जो कि ने एक कर करो करें बताइए बताओ दिखाओ दिखाइए""".split()),
    "hg": set("""ka ki ke ko se mein par tak aur ya bhi hi to toh hai hain tha thi the ho yeh woh ye
        wo is us in un jo ki ne ek kar karo batao dikhao do""".split()),
}


def remove_stopwords(words, lang):
    sw = STOPWORDS.get(lang, set())
    # normalize Hindi stopwords the same way the input was normalized
    if lang == "hi":
        sw = {normalize_unicode(w) for w in sw}
    return [w for w in words if w.lower() not in sw]


# ---------------------------------------------------------------------------
# 11. Stemming
# ---------------------------------------------------------------------------
_porter = PorterStemmer()

# Lightweight Hindi stemmer suffixes (Ramanathan & Rao, 2003), longest first
HINDI_SUFFIXES = sorted("""ाएंगी ाएंगे ाऊंगी ाऊंगा ाइयां ाइयों ाइयाँ ाएगी ाएगा ाओगी ाओगे एंगी ेंगी एंगे ेंगे ूंगी ूंगा
    ातीं नाओं नाएं ताओं ताएं ियाँ ियों ियां ाकर ाइए ाईं ाया ेगी ेगा ोगी ोगे ाने ाना ाते ाती ाता तीं ाओं
    ाएं ुओं ुएं ुआं कर ाओ िए ाई ाए ने नी ना ते ीं ती ता ाँ ां ों ें ो े ू ु ी ि ा""".split(), key=len, reverse=True)
HINGLISH_SUFFIXES = ["iyon", "iyan", "aon", "on", "ein", "en", "ne", "na", "ni", "te", "ta", "ti"]


def stem_word(word, word_lang):
    if word_lang == "EN":
        return _porter.stem(word)
    if word_lang == "HI":
        word = normalize_unicode(word)
        for suf in HINDI_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 2:
                return word[: -len(suf)]
        return word
    if word_lang == "HI-ROM":
        for suf in HINGLISH_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 4:
                return word[: -len(suf)]
        return word
    return word


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess(text, lang=None, drop_stopwords=False, normalize_spelling=True):
    """Run every step and return a dict with the intermediate results.

    `tokens` in the result is the final list the later phases use
    (normalized, numbers as digits, stopwords kept unless drop_stopwords=True).
    """
    steps = {"original": text}
    norm = normalize_unicode(text)
    steps["normalized"] = norm
    low = lowercase(norm)
    steps["lowercased"] = low
    steps["sentences"] = split_sentences(low)

    toks = tokenize(low)
    steps["tokens"] = [t for t, _ in toks]
    steps["scripts"] = tag_scripts(toks)

    lang = lang or detect_language(toks)
    steps["language"] = lang
    steps["word_langs"] = tag_word_language(toks, lang)

    words = [t for t, s in toks if s != "PUNCT"]
    word_langs = [wl for (t, s), wl in zip(toks, steps["word_langs"]) if s != "PUNCT"]
    if lang == "hg" and normalize_spelling:
        words, changes = normalize_hinglish(words)
    else:
        changes = []
    steps["hinglish_normalized"] = words
    steps["spelling_changes"] = [{"from": a, "to": b, "rule": r} for a, b, r in changes]

    words = numbers_to_digits(words, lang)
    steps["numbers"] = words

    no_stop = remove_stopwords(words, lang)
    steps["without_stopwords"] = no_stop
    steps["stems"] = [stem_word(w, wl) for w, wl in zip(words, word_langs)]

    steps["final"] = no_stop if drop_stopwords else words
    return steps


def tokens_for_model(text, lang=None):
    """Shortcut used by the n-gram and HMM phases: normalized tokens, punctuation removed."""
    return preprocess(text, lang)["final"]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in ["How many students are in the Computer department?",
              "कंप्यूटर विभाग में कितने छात्र हैं? पाँच से ज़्यादा।",
              "computer dept me kitney studnts h?",
              "top paanch students ka naam btao jinka cgpa 8.5 se jyada hai"]:
        r = preprocess(s)
        print(r["language"], r["tokens"], r["word_langs"], r["spelling_changes"], r["final"], r["stems"], sep="\n  ")
