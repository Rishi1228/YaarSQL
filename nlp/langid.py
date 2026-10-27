"""Phase 2 — Language identification with character n-gram models.

One character-trigram language model per language (English, Hindi, Hinglish), add-k smoothed.
A sentence is assigned to the language whose model gives it the highest log-probability.
Character n-grams work well here because Hinglish has tell-tale letter patterns
("ain", "kya", "hai", "aa") even when it mixes in English words.
"""
import math
from collections import Counter, defaultdict
from functools import lru_cache

from nlp.corpora import LANGS, corpus, raw_texts
from nlp.preprocess import normalize_unicode


class CharNGramLM:
    def __init__(self, n=3, k=0.5):
        self.n, self.k = n, k

    def _pad(self, text):
        return " " * (self.n - 1) + text + " "

    def fit(self, texts):
        self.counts = defaultdict(Counter)
        chars = set()
        for t in texts:
            s = self._pad(t)
            chars.update(s)
            for i in range(self.n - 1, len(s)):
                self.counts[s[i - self.n + 1:i]][s[i]] += 1
        self.totals = {h: sum(c.values()) for h, c in self.counts.items()}
        self.chars = chars
        return self

    def logprob(self, text, V):
        s, lp = self._pad(text), 0.0
        for i in range(self.n - 1, len(s)):
            h, c = s[i - self.n + 1:i], s[i]
            num = self.counts[h][c] if h in self.counts else 0
            lp += math.log((num + self.k) / (self.totals.get(h, 0) + self.k * V))
        return lp


class LanguageIdentifier:
    def __init__(self, n=3, k=0.5):
        self.n, self.k = n, k

    def fit(self, texts_by_lang):
        self.models = {lang: CharNGramLM(self.n, self.k).fit(texts) for lang, texts in texts_by_lang.items()}
        self.V = len(set().union(*(m.chars for m in self.models.values()))) + 1
        return self

    def scores(self, text):
        """Log-probability per language, plus normalized posterior probabilities."""
        text = normalize_unicode(text).lower()
        logp = {lang: m.logprob(text, self.V) for lang, m in self.models.items()}
        best = max(logp.values())
        z = sum(math.exp(v - best) for v in logp.values())
        post = {lang: math.exp(v - best) / z for lang, v in logp.items()}
        per_char = {lang: v / (len(text) + 1) for lang, v in logp.items()}
        return {"logp": logp, "posterior": post, "per_char": per_char}

    def predict(self, text):
        s = self.scores(text)["logp"]
        return max(s, key=s.get)


def training_texts():
    """Our training questions, plus general text so the models are not only about databases."""
    texts = {lang: raw_texts(lang, "train") for lang in LANGS}
    texts["en"] += [" ".join(s) for s in corpus("brown-news")[0][:1500]]
    texts["hi"] += [" ".join(s) for s in corpus("hindi-news")[0]]
    try:  # Hinglish POS corpus (Phase 3) adds everyday Hinglish sentences
        from nlp.hmm_pos import hinglish_sentences
        texts["hg"] += [" ".join(w for w, _ in s) for s in hinglish_sentences()]
    except Exception:
        pass
    return texts


@lru_cache(maxsize=None)
def get_identifier(n=3):
    return LanguageIdentifier(n=n).fit(training_texts())
