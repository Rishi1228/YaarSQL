"""Cached n-gram models and the JSON payloads behind the N-gram Lab page."""
import json
import math
from functools import lru_cache
from pathlib import Path

from nlp.corpora import CORPORA, corpus
from nlp.intent import get_classifier
from nlp.langid import get_identifier
from nlp.ngrams import NGramModel, SMOOTHINGS
from nlp.preprocess import preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _num(x):
    """JSON-safe float (inf -> None)."""
    return None if x is None or math.isinf(x) or math.isnan(x) else x


@lru_cache(maxsize=64)
def get_model(corpus_name, n, smoothing):
    return NGramModel(n, smoothing).fit(corpus(corpus_name)[0])


def overview(corpus_name, n, smoothing):
    model = get_model(corpus_name, n, smoothing)
    top = {}
    for order in range(1, n + 1):
        top[order] = [{"gram": " ".join(g), "count": c} for g, c in model.top_ngrams(order, 15)]
    return {
        "corpus": CORPORA[corpus_name],
        "stats": model.stats(),
        "top": top,
        "matrix": bigram_matrix(corpus_name, smoothing),
        "generated": [" ".join(model.generate(max_len=18, seed=s)) for s in range(5)],
        "test_perplexity": _num(model.perplexity(corpus(corpus_name)[1])),
    }


def bigram_matrix(corpus_name, smoothing, size=7):
    """P(column word | row word) for the most frequent words — the classic bigram table."""
    model = get_model(corpus_name, 2, smoothing)
    words = model.most_common_words(size)
    counts = [[model.counts[2][(a,)][b] for b in words] for a in words]
    probs = [[model.prob(b, [a]) for b in words] for a in words]
    return {"words": words, "counts": counts, "probs": probs}


def perplexity_table(corpus_name):
    """Held-out perplexity for N = 1..4 under every smoothing method (cached on disk)."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"perplexity_{corpus_name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    train, test = corpus(corpus_name)
    table = {}
    for s in SMOOTHINGS:
        table[s] = [_num(NGramModel(n, s).fit(train).perplexity(test)) for n in (1, 2, 3, 4)]
    out = {"orders": [1, 2, 3, 4], "table": table, "labels": SMOOTHINGS, "test_sentences": len(test)}
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def sentence(corpus_name, n, smoothing, text):
    model = get_model(corpus_name, n, smoothing)
    lang = "hi" if corpus_name.endswith("hi") or corpus_name == "hindi-news" else \
        "hg" if corpus_name.endswith("hg") else "en"
    tokens = preprocess(text, lang)["final"]
    rows, logp = model.sentence_breakdown(tokens)
    for r in rows:
        r["logp"] = _num(r["logp"])
    count = len(tokens) + 1
    ppl = math.exp(-logp / count) if logp != float("-inf") else None
    return {"tokens": tokens, "rows": rows, "logprob": _num(logp),
            "prob": math.exp(logp) if logp != float("-inf") else 0.0, "perplexity": ppl}


# --------------------------------------------------------------------------- uses in the project
@lru_cache(maxsize=3)
def autocomplete_model(lang):
    """Trigram Kneser-Ney model over the training questions of one language."""
    return NGramModel(3, "kneser_ney").fit(corpus(f"questions-{lang}")[0])


def autocomplete(text, lang=None, limit=6):
    lang = lang or (get_identifier().predict(text) if text.strip() else "en")
    ends_with_space = text.endswith(" ") or not text
    tokens = preprocess(text, lang)["final"] if text.strip() else []
    prefix = ""
    if not ends_with_space and tokens:
        prefix, tokens = tokens[-1], tokens[:-1]
    model = autocomplete_model(lang)
    sugg = model.next_words(tokens, limit=limit, prefix=prefix)
    return {"lang": lang, "context": tokens[-2:], "prefix": prefix,
            "suggestions": [{"word": w, "p": p} for w, p in sugg]}


def langid(text):
    s = get_identifier().scores(text)
    return {"prediction": max(s["logp"], key=s["logp"].get), **s}


def intent(text):
    lid = get_identifier().predict(text)
    tokens = preprocess(text, lid)["final"]
    clf = get_classifier()
    return {"lang": lid, "tokens": tokens, "features": clf.features(tokens), "ranking": clf.rank(tokens)[:5]}
