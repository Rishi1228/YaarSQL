import json
from pathlib import Path

from nlp.preprocess import (edit_distance, normalize_hinglish, normalize_unicode, preprocess,
                            split_sentences, stem_word, tokenize)

ROOT = Path(__file__).resolve().parent.parent


def test_unicode_normalization_folds_hindi_variants():
    assert normalize_unicode("ज़्यादा") == normalize_unicode("ज्यादा")
    assert normalize_unicode("पाँच") == "पांच"
    assert normalize_unicode("१२३") == "123"


def test_sentence_split_handles_danda():
    assert split_sentences("यह पहला है। यह दूसरा है।") == ["यह पहला है।", "यह दूसरा है।"]


def test_tokenizer_keeps_decimals_and_hindi_words():
    toks = tokenize("cgpa 8.5 से ज़्यादा?")
    assert [t for t, _ in toks] == ["cgpa", "8.5", "से", "ज़्यादा", "?"]
    assert [s for _, s in toks] == ["LATIN", "NUM", "DEVANAGARI", "DEVANAGARI", "PUNCT"]


def test_edit_distance():
    assert edit_distance("kitne", "kitne") == 0
    assert edit_distance("kitney", "kitne") == 1
    assert edit_distance("sunday", "saturday") == 3


def test_hinglish_variants_normalized():
    words, changes = normalize_hinglish(["kitney", "studnts", "h"])
    assert words == ["kitne", "students", "hain"]
    assert len(changes) == 3


def test_language_detection():
    assert preprocess("How many students are there?")["language"] == "en"
    assert preprocess("कितने छात्र हैं?")["language"] == "hi"
    assert preprocess("kitne students hain?")["language"] == "hg"
    # "me" is both English and a Hinglish spelling of "mein"
    assert preprocess("tell me the number of students")["language"] == "en"


def test_language_detection_on_whole_dataset():
    data = []
    for name in ("train", "test"):
        data += json.loads((ROOT / "data" / "dataset" / f"{name}.json").read_text(encoding="utf-8"))
    correct = sum(preprocess(it["question"])["language"] == it["lang"] for it in data)
    assert correct / len(data) > 0.95


def test_number_words():
    assert "5" in preprocess("top paanch students")["final"]
    assert "5" in preprocess("शीर्ष पाँच छात्र")["final"]
    assert "3" in preprocess("top three students")["final"]


def test_code_mixed_word_tags():
    r = preprocess("computer department mein kitne students hain")
    assert r["word_langs"] == ["EN", "EN", "HI-ROM", "HI-ROM", "EN", "HI-ROM"]


def test_stemming():
    assert stem_word("running", "EN") == "run"
    assert stem_word("छात्रों", "HI") == "छात्र"
    assert stem_word("shikshakon", "HI-ROM") == "shikshak"
    assert stem_word("kitne", "HI-ROM") == "kitne"  # too short to strip


def test_english_abbreviation_in_hinglish_is_english():
    r = preprocess("computer dept me kitney studnts h?")
    assert r["word_langs"][:2] == ["EN", "EN"]
    assert r["final"] == ["computer", "department", "mein", "kitne", "students", "hain"]
