import math

import pytest

from nlp.intent import evaluate as eval_intent
from nlp.langid import get_identifier
from nlp.ngrams import EOS, NGramModel, SMOOTHINGS
from nlp.corpora import dataset
from nlp.schema import link, mask

TOY = [["i", "like", "tea"], ["i", "like", "coffee"], ["you", "like", "tea"]]


def test_counts_and_mle():
    m = NGramModel(2, "mle").fit(TOY)
    assert m.counts[2][("i",)]["like"] == 2
    assert m.prob("like", ["i"]) == 1.0
    assert m.prob("tea", ["like"]) == pytest.approx(2 / 3)
    assert m.prob("i", []) == pytest.approx(2 / 3)  # P(i | <s>)


def test_laplace_formula():
    m = NGramModel(2, "laplace").fit(TOY)
    # vocab = i like tea coffee you </s> <unk> = 7
    assert m.V == 7
    assert m.prob("coffee", ["like"]) == pytest.approx((1 + 1) / (3 + 7))


@pytest.mark.parametrize("smoothing", [s for s in SMOOTHINGS if s not in ("mle", "backoff")])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_distributions_sum_to_one(smoothing, n):
    m = NGramModel(n, smoothing).fit(TOY)
    for ctx in ([], ["i"], ["i", "like"], ["unseen", "words"]):
        assert sum(m.prob(w, ctx) for w in m.vocab) == pytest.approx(1.0)


def test_mle_perplexity_infinite_on_unseen_and_kn_finite():
    test = [["i", "like", "milk"]]
    assert NGramModel(2, "mle").fit(TOY).perplexity(test) == math.inf
    assert NGramModel(2, "kneser_ney").fit(TOY).perplexity(test) < math.inf


def test_bigram_beats_unigram_on_questions():
    from nlp.corpora import corpus
    train, test = corpus("questions-hg")
    uni = NGramModel(1, "kneser_ney").fit(train).perplexity(test)
    bi = NGramModel(2, "kneser_ney").fit(train).perplexity(test)
    assert bi < uni


def test_generation_and_autocomplete():
    m = NGramModel(2, "laplace").fit(TOY)
    s = m.generate(seed=1)
    assert all(w in m.vocab for w in s)
    assert m.next_words(["i"], limit=1)[0][0] == "like"
    assert [w for w, _ in m.next_words([], prefix="co")] == ["coffee"]
    assert EOS not in m.generate(seed=2)


def test_language_identification():
    lid = get_identifier()
    assert lid.predict("show me all students") == "en"
    assert lid.predict("kitne students hain") == "hg"
    assert lid.predict("सभी छात्र दिखाओ") == "hi"
    items = dataset("test")
    acc = sum(lid.predict(it["question"]) == it["lang"] for it in items) / len(items)
    assert acc > 0.95


def test_schema_linking_and_masking():
    toks = ["pune", "ke", "kitne", "students", "hain"]
    kinds = [(m["type"], m["key"], m["value"]) for m in link(toks)]
    assert ("value", "students.city", "Pune") in kinds
    assert ("table", "students", None) in kinds
    assert mask(toks) == ["<CITY>", "ke", "kitne", "<TABLE>", "hain"]
    # English "it" is not the IT department unless a department word follows
    assert link(["is", "it", "the", "highest"]) == []
    assert link(["it", "department"])[0]["value"] == "IT"
    # fuzzy match: misspelled column
    assert link(["salry"], fuzzy=True)[0]["key"] == "faculty.salary"


def test_intent_top3_accuracy():
    acc, _ = eval_intent()
    assert acc["all"]["top3"] > 0.85
