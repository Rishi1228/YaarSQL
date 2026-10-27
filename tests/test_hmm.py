import itertools
import math

import pytest

from nlp.hmm_pos import (IIIT_TO_UNIVERSAL, HMMTagger, END, START, evaluate, get_tagger, hindi_sentences,
                         hinglish_sentences, splits)
from nlp.translit import transliterate

TOY = [
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
    [("a", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("the", "DET"), ("runs", "NOUN"), ("stop", "VERB")],
    [("dogs", "NOUN"), ("run", "VERB")],
]


def brute_force(tagger, words):
    """Score every possible tag sequence; return the best one and log of the summed probability."""
    best, best_score, total = None, -math.inf, []
    for seq in itertools.product(tagger.tags, repeat=len(words)):
        score = tagger.log_trans[START][seq[0]] + tagger.log_trans[seq[-1]][END]
        for i, (w, t) in enumerate(zip(words, seq)):
            score += tagger.log_emissions(w)[0][t]
            if i:
                score += tagger.log_trans[seq[i - 1]][t]
        total.append(score)
        if score > best_score:
            best, best_score = list(seq), score
    m = max(total)
    return best, best_score, m + math.log(sum(math.exp(s - m) for s in total))


@pytest.mark.parametrize("words", [["the", "dog", "runs"], ["a", "runs", "stop"], ["dogs", "zebra", "run"]])
def test_viterbi_matches_brute_force(words):
    tagger = HMMTagger().fit(TOY)
    path, trellis = tagger.viterbi(words)
    best, best_score, log_total = brute_force(tagger, words)
    assert path == best
    assert trellis["best"] == pytest.approx(best_score)
    assert tagger.forward(words) == pytest.approx(log_total)
    assert tagger.forward(words) >= trellis["best"]


def test_transition_rows_sum_to_one():
    tagger = HMMTagger().fit(TOY)
    for row in tagger.transition_matrix()["p"]:
        assert sum(row) == pytest.approx(1.0)


def test_numbers_and_unknown_words():
    tagger = get_tagger("en")
    tags = dict(tagger.tag(["show", "students", "with", "cgpa", "above", "8.5"]))
    assert tags["8.5"] == "NUM"
    assert tags["with"] == "ADP"


def test_english_and_hindi_accuracy():
    for lang, threshold in [("en", 0.93), ("hi", 0.90)]:
        train, test = splits(lang)
        assert evaluate(get_tagger(lang), test)["accuracy"] > threshold


def test_hinglish_tagger():
    tagger = get_tagger("hg")
    tags = [t for _, t in tagger.tag("8 se zyada cgpa wale students dikhao".split())]
    assert tags == ["NUM", "ADP", "ADJ", "NOUN", "PRT", "NOUN", "VERB"]
    assert len(hinglish_sentences()) >= 100


def test_iiit_mapping_covers_corpus():
    tags = {t for s in hindi_sentences("iiit") for _, t in s}
    assert tags - {"X"} <= set(IIIT_TO_UNIVERSAL)


def test_transliteration():
    assert transliterate("कितने छात्र हैं") == "kitne chhatra hain"
    assert transliterate("कमल समय") == "kamal samay"
    assert transliterate("लड़की") == "ladki"
