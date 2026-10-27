"""Phase 2 — Intent (question type) detection with n-gram language models.

One n-gram LM is trained per intent (COUNT_ALL, MAX_VAL, GROUP_COUNT, ...) on the training
questions of all three languages, after cell values are masked (<CITY>, <DEPT>, <NUM>, <YEAR>).
A new question goes to the intent whose model gives it the highest probability:

    intent* = argmax_i  log P(intent_i) + log P_i(w_1 ... w_m)

With n = 1 and Laplace smoothing this is exactly multinomial Naive Bayes, which we use as the baseline.
"""
import math
from collections import Counter
from functools import lru_cache

from nlp.corpora import question_tokens
from nlp.ngrams import NGramModel
from nlp.preprocess import STOPWORDS, normalize_unicode
from nlp.schema import mask

# Function words carry little information about the question type; dropping them lets the
# key words (how many / sabse zyada / औसत) dominate the score.
FUNCTION_WORDS = {normalize_unicode(w) for ws in STOPWORDS.values() for w in ws}


class NGramIntentClassifier:
    def __init__(self, n=2, smoothing="interpolation", drop_stopwords=True):
        self.n, self.smoothing, self.drop_stopwords = n, smoothing, drop_stopwords

    def features(self, tokens, pos=None, mentions=None):
        m = mask(tokens, pos, mentions)
        return [w for w in m if w not in FUNCTION_WORDS] if self.drop_stopwords else m

    def fit(self, examples):
        """examples: list of (tokens, intent)"""
        by_intent = {}
        for toks, intent in examples:
            by_intent.setdefault(intent, []).append(self.features(toks))
        total = len(examples)
        vocab = {w for sents in by_intent.values() for s in sents for w in s}
        self.prior = {i: math.log(len(s) / total) for i, s in by_intent.items()}
        self.models = {i: NGramModel(self.n, self.smoothing).fit(s, vocab=vocab) for i, s in by_intent.items()}
        return self

    def rank(self, tokens, pos=None, mentions=None, per_token=True):
        """All intents sorted by score, with softmax-normalized probabilities.

        Whole-sentence likelihoods differ by many orders of magnitude, which makes the softmax
        overconfident. With per_token=True the log-probabilities are divided by the sentence
        length first (a temperature), so the probabilities stay comparable with other evidence.
        """
        m = self.features(tokens, pos, mentions)
        scores = {i: self.prior[i] + model.sentence_logprob(m) for i, model in self.models.items()}
        temp = (len(m) + 1) if per_token else 1.0
        best = max(scores.values())
        z = sum(math.exp((s - best) / temp) for s in scores.values())
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [{"intent": i, "logp": s, "prob": math.exp((s - best) / temp) / z} for i, s in ranked]

    def predict(self, tokens):
        return self.rank(tokens)[0]["intent"]


@lru_cache(maxsize=None)
def get_classifier(n=2, smoothing="interpolation", drop_stopwords=True, holdout_template=None):
    """holdout_template=3 leaves out every 4th phrasing so it can serve as a validation set."""
    examples = [(toks, it["intent"]) for it, toks in question_tokens("train")
                if holdout_template is None or not it["template"].endswith(f"/{holdout_template}")]
    return NGramIntentClassifier(n, smoothing, drop_stopwords).fit(examples)


def evaluate(n=2, smoothing="interpolation", drop_stopwords=True, split="test"):
    """Top-1 and top-3 accuracy per language, plus the confusion counts."""
    clf = get_classifier(n, smoothing, drop_stopwords)
    top1, top3, totals, confusion = Counter(), Counter(), Counter(), Counter()
    for it, toks in question_tokens(split):
        ranked = [r["intent"] for r in clf.rank(toks)]
        lang = it["lang"]
        totals[lang] += 1
        top1[lang] += ranked[0] == it["intent"]
        top3[lang] += it["intent"] in ranked[:3]
        confusion[(it["intent"], ranked[0])] += 1
    out = {lang: {"top1": top1[lang] / totals[lang], "top3": top3[lang] / totals[lang]} for lang in totals}
    n_all = max(1, sum(totals.values()))
    out["all"] = {"top1": sum(top1.values()) / n_all, "top3": sum(top3.values()) / n_all}
    return out, confusion
