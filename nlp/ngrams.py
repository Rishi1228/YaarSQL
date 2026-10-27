"""Phase 2 — N-gram language models, written from scratch.

NGramModel(n, smoothing) supports any n >= 1 (unigram, bigram, trigram, 4-gram ...).

Smoothing methods
    mle           maximum likelihood: c(h,w) / c(h)                     (unseen -> probability 0)
    laplace       add-one:  (c(h,w) + 1) / (c(h) + V)
    addk          add-k:    (c(h,w) + k) / (c(h) + kV)
    interpolation linear interpolation of MLE estimates of every order (+ add-one unigram)
    backoff       "stupid backoff" (Brants et al., 2007): a score, not a true probability
    kneser_ney    interpolated Kneser-Ney with absolute discount d = 0.75

Sentences are padded with n-1 start symbols <s> and one end symbol </s>.
Words never seen in training are mapped to <unk>.
"""
import math
import random
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
SMOOTHINGS = {
    "mle": "MLE (no smoothing)",
    "laplace": "Laplace (add-1)",
    "addk": "Add-k",
    "interpolation": "Linear interpolation",
    "backoff": "Stupid backoff",
    "kneser_ney": "Kneser-Ney",
}


class NGramModel:
    def __init__(self, n=2, smoothing="laplace", k=0.1, discount=0.75, lambdas=None, backoff_alpha=0.4):
        if n < 1:
            raise ValueError("n must be >= 1")
        if smoothing not in SMOOTHINGS:
            raise ValueError(f"unknown smoothing {smoothing!r}")
        self.n, self.smoothing = n, smoothing
        self.k, self.d, self.alpha = k, discount, backoff_alpha
        # interpolation weights for orders 1..n (higher orders weigh more)
        self.lambdas = lambdas or [i / sum(range(1, n + 1)) for i in range(1, n + 1)]

    # ------------------------------------------------------------------ training
    def fit(self, sentences, vocab=None):
        """sentences: iterable of token lists.

        vocab: optional shared vocabulary (used when several models must be comparable,
        e.g. one model per intent): every model then gives unseen words the same treatment.
        """
        n = self.n
        # counts[k][context][word], context is a tuple of k-1 words
        self.counts = {k: defaultdict(Counter) for k in range(1, n + 1)}
        self.vocab = {EOS, UNK}
        self.num_sentences = 0
        for sent in sentences:
            self.num_sentences += 1
            self.vocab.update(sent)
            seq = [BOS] * (n - 1) + list(sent) + [EOS]
            for i in range(n - 1, len(seq)):
                for k in range(1, n + 1):
                    self.counts[k][tuple(seq[i - k + 1:i])][seq[i]] += 1
        if vocab is not None:
            self.vocab |= set(vocab)
        self.vocab.discard(BOS)
        self.V = len(self.vocab)
        self.totals = {k: {h: sum(c.values()) for h, c in self.counts[k].items()} for k in self.counts}
        self.N = self.totals[1][()]  # number of training tokens (incl. </s>)
        if self.smoothing == "kneser_ney":
            self._fit_kneser_ney()
        return self

    def _fit_kneser_ney(self):
        # continuation counts: in how many different contexts does (h, w) appear?
        self.cont = {k: defaultdict(Counter) for k in range(1, self.n)}
        for k in range(2, self.n + 1):
            for h, ctr in self.counts[k].items():
                for w in ctr:
                    self.cont[k - 1][h[1:]][w] += 1
        self.cont_totals = {k: {h: sum(c.values()) for h, c in self.cont[k].items()} for k in self.cont}

    # ------------------------------------------------------------------ probabilities
    def _ctx(self, context):
        """Last n-1 words of the context, padded with <s>, unknown words -> <unk>."""
        context = [w if (w in self.vocab or w == BOS) else UNK for w in context]
        context = [BOS] * (self.n - 1) + context
        return tuple(context[len(context) - (self.n - 1):]) if self.n > 1 else ()

    def prob(self, word, context=()):
        """P(word | context). `context` is the list of preceding words."""
        if word not in self.vocab:
            word = UNK
        h = self._ctx(context)
        s = self.smoothing
        if s == "mle":
            return self._mle(word, h)
        if s == "laplace":
            return self._addk(word, h, 1.0)
        if s == "addk":
            return self._addk(word, h, self.k)
        if s == "interpolation":
            # only orders whose context was seen take part; their weights are renormalized
            parts = [(self.lambdas[0], self._addk(word, (), 1.0))]
            for k in range(2, self.n + 1):
                hk = h[len(h) - (k - 1):]
                if self.totals[k].get(hk):
                    parts.append((self.lambdas[k - 1], self._mle(word, hk)))
            z = sum(lam for lam, _ in parts)
            return sum(lam * p for lam, p in parts) / z
        if s == "backoff":
            return self._stupid_backoff(word, h)
        return self._kn(word, h, top=True)

    def _mle(self, word, h):
        total = self.totals[len(h) + 1].get(h, 0)
        return self.counts[len(h) + 1][h][word] / total if total else 0.0

    def _addk(self, word, h, k):
        order = len(h) + 1
        c = self.counts[order][h][word] if h in self.counts[order] else 0
        return (c + k) / (self.totals[order].get(h, 0) + k * self.V)

    def _stupid_backoff(self, word, h):
        factor = 1.0
        while h:
            order = len(h) + 1
            if h in self.counts[order] and self.counts[order][h][word] > 0:
                return factor * self.counts[order][h][word] / self.totals[order][h]
            factor *= self.alpha
            h = h[1:]
        return factor * (self.counts[1][()][word] + 1) / (self.N + self.V)

    def _kn(self, word, h, top):
        """Interpolated Kneser-Ney. Highest order uses real counts, lower orders continuation counts."""
        order = len(h) + 1
        if top:
            table, totals = self.counts[order], self.totals[order]
        else:
            table, totals = self.cont[order], self.cont_totals[order]
        total = totals.get(h, 0)
        if order == 1:
            ctr = table[()]
            lam = self.d * len(ctr) / total
            return max(ctr[word] - self.d, 0) / total + lam / self.V
        if not total:
            return self._kn(word, h[1:], top=False)
        ctr = table[h]
        lam = self.d * len(ctr) / total
        return max(ctr[word] - self.d, 0) / total + lam * self._kn(word, h[1:], top=False)

    # ------------------------------------------------------------------ evaluation
    def sentence_breakdown(self, tokens):
        """Probability of every word in a sentence. Returns (rows, total_logprob)."""
        rows, logp, history = [], 0.0, []
        for w in list(tokens) + [EOS]:
            p = self.prob(w, history)
            lp = math.log(p) if p > 0 else float("-inf")
            logp += lp
            known = w in self.vocab
            rows.append({"word": w, "context": list(self._ctx(history)), "p": p, "logp": lp, "known": known})
            history.append(w)
        return rows, logp

    def sentence_logprob(self, tokens):
        return self.sentence_breakdown(tokens)[1]

    def perplexity(self, sentences):
        """exp(-1/N * sum log P). Returns inf if any word gets probability 0 (MLE)."""
        logp, count = 0.0, 0
        for s in sentences:
            lp = self.sentence_logprob(s)
            if lp == float("-inf"):
                return float("inf")
            logp += lp
            count += len(s) + 1
        return math.exp(-logp / count) if count else float("inf")

    # ------------------------------------------------------------------ inspection
    def top_ngrams(self, order, limit=20, skip_boundaries=True):
        rows = []
        for h, ctr in self.counts[order].items():
            for w, c in ctr.items():
                gram = h + (w,)
                if skip_boundaries and (BOS in gram or EOS in gram):
                    continue
                rows.append((gram, c))
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows[:limit]

    def most_common_words(self, limit=10):
        ctr = self.counts[1][()]
        return [w for w, _ in ctr.most_common() if w not in (EOS, UNK)][:limit]

    def candidates(self, context):
        """Words seen after the longest matching context (falls back to shorter ones)."""
        h = self._ctx(context)
        while True:
            ctr = self.counts[len(h) + 1].get(h)
            if ctr:
                return list(ctr)
            if not h:
                return [w for w in self.vocab if w != UNK]
            h = h[1:]

    def next_words(self, context, limit=5, prefix=""):
        """Most likely next words (used for autocomplete)."""
        pool = self.candidates(context) if not prefix else [w for w in self.vocab if w.startswith(prefix)]
        pool = [w for w in pool if w not in (UNK,) and (w != EOS or not prefix)]
        scored = sorted(((self.prob(w, context), w) for w in pool), reverse=True)
        return [(w, p) for p, w in scored[:limit]]

    def generate(self, max_len=20, seed=None, start=()):
        """Sample a sentence word by word from P(w | previous words)."""
        rng = random.Random(seed)
        out = list(start)
        while len(out) < max_len:
            pool = [w for w in self.candidates(out) if w != UNK]
            weights = [self.prob(w, out) for w in pool]
            if not pool or sum(weights) == 0:
                break
            w = rng.choices(pool, weights=weights)[0]
            if w == EOS:
                break
            out.append(w)
        return out

    def stats(self):
        return {
            "n": self.n,
            "sentences": self.num_sentences,
            "tokens": self.N - self.num_sentences,  # without </s>
            "vocab": self.V - 2,                     # without </s>, <unk>
            "distinct": {k: sum(len(c) for c in self.counts[k].values()) for k in self.counts},
        }
