"""Loads and tokenizes every corpus the project uses (cached, so each is processed once)."""
import json
import random
import re
from functools import lru_cache
from pathlib import Path

import nltk

from nlp.preprocess import normalize_unicode, tokens_for_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
nltk.data.path.insert(0, str(DATA / "nltk_data"))

LANGS = {"en": "English", "hi": "Hindi", "hg": "Hinglish"}

CORPORA = {
    "questions-en": "Questions — English (our dataset)",
    "questions-hi": "Questions — Hindi (our dataset)",
    "questions-hg": "Questions — Hinglish (our dataset)",
    "brown-news": "Brown corpus — news (English)",
    "hindi-news": "NLTK Indian corpus — Hindi news",
}


@lru_cache(maxsize=None)
def dataset(split):
    """Question-SQL pairs: split = train | test | wild."""
    path = DATA / "dataset" / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@lru_cache(maxsize=None)
def question_tokens(split):
    """[(item, tokens)] with tokens from the Phase 1 preprocessing pipeline."""
    return [(it, tokens_for_model(it["question"], it["lang"])) for it in dataset(split)]


def _split(sents, test_ratio=0.1, seed=0):
    idx = list(range(len(sents)))
    random.Random(seed).shuffle(idx)
    cut = int(len(sents) * (1 - test_ratio))
    return [sents[i] for i in idx[:cut]], [sents[i] for i in idx[cut:]]


@lru_cache(maxsize=None)
def corpus(name):
    """Returns (train_sentences, test_sentences) as lists of token lists."""
    if name.startswith("questions-"):
        lang = name.split("-")[1]
        train = [t for it, t in question_tokens("train") if it["lang"] == lang]
        test = [t for it, t in question_tokens("test") if it["lang"] == lang]
        return train, test
    if name == "brown-news":
        from nltk.corpus import brown
        sents = [[w.lower() for w in s if re.fullmatch(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", w)]
                 for s in brown.sents(categories="news")]
        return _split([s for s in sents if s])
    if name == "hindi-news":
        from nltk.corpus import indian
        sents = [[normalize_unicode(w) for w in s if re.search(r"[ऀ-ॿ\d]", w)]
                 for s in indian.sents("hindi.pos")]
        return _split([s for s in sents if s])
    raise KeyError(name)


def raw_texts(lang, split="train"):
    """Plain (normalized, lowercased) question strings for one language."""
    return [normalize_unicode(it["question"]).lower() for it in dataset(split) if it["lang"] == lang]
