"""Phase 3 — Part-of-speech tagging with a Hidden Markov Model, written from scratch.

    hidden states  = POS tags          observations = words
    P(tags, words) = Π P(t_i | t_{i-1}) · P(w_i | t_i)        (first-order / bigram HMM)

Training (supervised, from a tagged corpus):
    transition  P(t | t_prev) = (c(t_prev, t) + α) / (c(t_prev) + α·|T|)       add-α smoothing
    emission    P(w | t)      = (c(t, w) + β) / (c(t) + β·(V+1))
Unknown words: a suffix model (as in the TnT tagger, Brants 2000) estimates P(t | last letters)
from rare training words; numbers are forced to NUM and punctuation to ".".

Decoding: the Viterbi algorithm in log space (the full trellis is returned so the website can draw it).
The forward algorithm gives P(words) summed over all tag sequences.

Corpora
    English  : Penn Treebank sample (NLTK), mapped to the 12-tag universal tagset
    Hindi    : NLTK Indian corpus hindi.pos (IIIT tagset), also mapped to universal tags
    Hinglish : data/pos/hinglish_gold.txt (hand-tagged), optionally combined with English and
               Romanized Hindi (the Hindi corpus passed through nlp.translit)
"""
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import nltk

from nlp.preprocess import normalize_unicode
from nlp.translit import transliterate_word

ROOT = Path(__file__).resolve().parent.parent
nltk.data.path.insert(0, str(ROOT / "data" / "nltk_data"))
MODELS_DIR = ROOT / "models"

START, END = "<S>", "</S>"
UNIVERSAL = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", ".", "X"]
TAG_INFO = {
    "NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb", "PRON": "pronoun",
    "DET": "determiner / quantifier", "ADP": "adposition (pre/postposition)", "NUM": "number",
    "CONJ": "conjunction", "PRT": "particle", ".": "punctuation", "X": "other",
}

# IIIT Hindi tagset (NLTK indian corpus) -> universal tags
IIIT_TO_UNIVERSAL = {
    "NN": "NOUN", "NNC": "NOUN", "NNP": "NOUN", "NNPC": "NOUN", "NLOC": "NOUN", "NVB": "NOUN",
    "VFM": "VERB", "VAUX": "VERB", "VRB": "VERB", "VNN": "VERB", "VJJ": "VERB",
    "JJ": "ADJ", "JVB": "ADJ", "QFNUM": "NUM", "QF": "DET", "PRP": "PRON", "QW": "PRON",
    "PREP": "ADP", "CC": "CONJ", "RB": "ADV", "INTF": "ADV", "RP": "PRT", "NEG": "PRT",
    "PUNC": ".", "SYM": ".", "": "X",
}

NUM_RE = re.compile(r"^\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?$")
PUNCT_RE = re.compile(r"^[^\wऀ-ॿ]+$")


# --------------------------------------------------------------------------- the model
class HMMTagger:
    def __init__(self, alpha=0.1, beta=0.001, max_suffix=4, rare=10):
        self.alpha, self.beta, self.max_suffix, self.rare = alpha, beta, max_suffix, rare

    def fit(self, tagged_sents):
        trans, emit, tag_count, word_count = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
        for sent in tagged_sents:
            prev = START
            for w, t in sent:
                trans[prev][t] += 1
                emit[t][w] += 1
                tag_count[t] += 1
                word_count[w] += 1
                prev = t
            trans[prev][END] += 1
        self.tags = sorted(tag_count)
        self.trans, self.emit, self.tag_count, self.word_count = trans, emit, tag_count, word_count
        self.V = len(word_count)
        self.word_tags = defaultdict(set)
        for t, ws in emit.items():
            for w in ws:
                self.word_tags[w].add(t)
        self._fit_suffixes()
        # precompute log transitions (incl. START row and END column)
        states = self.tags + [END]
        self.log_trans = {}
        for prev in [START] + self.tags:
            total = sum(trans[prev].values())
            self.log_trans[prev] = {
                t: math.log((trans[prev][t] + self.alpha) / (total + self.alpha * len(states))) for t in states
            }
        return self

    def _fit_suffixes(self):
        """P(tag | suffix) from rare words, smoothed by successive abstraction (TnT)."""
        rare_tags = Counter()
        self.suffix = defaultdict(Counter)
        for t, ws in self.emit.items():
            for w, c in ws.items():
                if self.word_count[w] <= self.rare:
                    rare_tags[t] += c
                    for k in range(1, min(self.max_suffix, len(w)) + 1):
                        self.suffix[w[-k:]][t] += c
        total = sum(rare_tags.values()) or 1
        self.p_rare = {t: (rare_tags[t] + 1) / (total + len(self.tags)) for t in self.tags}
        self.theta = statistics.pstdev(self.p_rare.values()) or 0.1
        total_tags = sum(self.tag_count.values())
        self.p_tag = {t: self.tag_count[t] / total_tags for t in self.tags}

    def _unknown_log_emissions(self, word):
        """log P(word | tag) for a word never seen in training (up to a constant)."""
        if NUM_RE.match(word):
            return {t: (0.0 if t == "NUM" else -20.0) for t in self.tags}
        if PUNCT_RE.match(word):
            return {t: (0.0 if t == "." else -20.0) for t in self.tags}
        p = dict(self.p_rare)
        for k in range(1, min(self.max_suffix, len(word)) + 1):
            ctr = self.suffix.get(word[-k:])
            if not ctr:
                break
            n = sum(ctr.values())
            p = {t: (ctr[t] + self.theta * p[t]) / (n + self.theta) for t in self.tags}
        # Bayes: P(w|t) ∝ P(t|suffix) / P(t)
        return {t: math.log(p[t] / max(self.p_tag[t], 1e-9)) - 12.0 for t in self.tags}

    def log_emissions(self, word):
        """{tag: log P(word | tag)} and whether the word was seen in training."""
        if word in self.word_count:
            return {t: math.log((self.emit[t][word] + self.beta) / (self.tag_count[t] + self.beta * (self.V + 1)))
                    for t in self.tags}, True
        return self._unknown_log_emissions(word), False

    # ------------------------------------------------------------------ decoding
    def viterbi(self, words):
        """Most likely tag sequence. Returns (tags, trellis) where trellis[i][tag] = best log score."""
        if not words:
            return [], {"scores": [], "back": [], "known": []}
        emissions = [self.log_emissions(w) for w in words]
        scores, back = [], []
        e0, _ = emissions[0]
        scores.append({t: self.log_trans[START][t] + e0[t] for t in self.tags})
        back.append({t: None for t in self.tags})
        for i in range(1, len(words)):
            e, _ = emissions[i]
            row, ptr = {}, {}
            for t in self.tags:
                best_prev = max(self.tags, key=lambda p: scores[i - 1][p] + self.log_trans[p][t])
                row[t] = scores[i - 1][best_prev] + self.log_trans[best_prev][t] + e[t]
                ptr[t] = best_prev
            scores.append(row)
            back.append(ptr)
        last = max(self.tags, key=lambda t: scores[-1][t] + self.log_trans[t][END])
        path = [last]
        for i in range(len(words) - 1, 0, -1):
            path.append(back[i][path[-1]])
        path.reverse()
        best = scores[-1][last] + self.log_trans[last][END]
        return path, {"scores": scores, "back": back, "known": [k for _, k in emissions], "best": best}

    def tag(self, words):
        return list(zip(words, self.viterbi(words)[0]))

    def forward(self, words):
        """log P(words), summing over every possible tag sequence."""
        def logsumexp(xs):
            m = max(xs)
            return m + math.log(sum(math.exp(x - m) for x in xs))
        if not words:
            return 0.0
        e = self.log_emissions(words[0])[0]
        alpha = {t: self.log_trans[START][t] + e[t] for t in self.tags}
        for w in words[1:]:
            e = self.log_emissions(w)[0]
            alpha = {t: logsumexp([alpha[p] + self.log_trans[p][t] for p in self.tags]) + e[t] for t in self.tags}
        return logsumexp([alpha[t] + self.log_trans[t][END] for t in self.tags])

    def transition_matrix(self):
        return {"rows": [START] + self.tags, "cols": self.tags + [END],
                "p": [[math.exp(self.log_trans[r][c]) for c in self.tags + [END]] for r in [START] + self.tags]}

    def top_emissions(self, tag, k=8):
        return [(w, c / self.tag_count[tag]) for w, c in self.emit[tag].most_common(k)]


class MostFrequentTagBaseline:
    """Tag each word with its most frequent training tag; unknown words get the most common tag."""

    def fit(self, tagged_sents):
        counts = defaultdict(Counter)
        tags = Counter()
        for s in tagged_sents:
            for w, t in s:
                counts[w][t] += 1
                tags[t] += 1
        self.best = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        self.default = tags.most_common(1)[0][0]
        return self

    def tag(self, words):
        return [(w, "NUM" if NUM_RE.match(w) else self.best.get(w, self.default)) for w in words]


# --------------------------------------------------------------------------- corpora
def _split(sents, test_ratio=0.2, seed=13):
    idx = list(range(len(sents)))
    random.Random(seed).shuffle(idx)
    cut = int(len(sents) * (1 - test_ratio))
    return [sents[i] for i in idx[:cut]], [sents[i] for i in idx[cut:]]


@lru_cache(maxsize=None)
def english_sentences():
    from nltk.corpus import treebank
    from nltk.tag.mapping import map_tag
    out = []
    for s in treebank.tagged_sents():
        sent = [(w.lower(), map_tag("en-ptb", "universal", t)) for w, t in s if t != "-NONE-"]
        if sent:
            out.append(sent)
    return out


@lru_cache(maxsize=None)
def hindi_sentences(tagset="universal"):
    from nltk.corpus import indian
    out = []
    for s in indian.tagged_sents("hindi.pos"):
        sent = [(normalize_unicode(w), IIIT_TO_UNIVERSAL.get(t, "X") if tagset == "universal" else (t or "X"))
                for w, t in s]
        if sent:
            out.append(sent)
    return out


@lru_cache(maxsize=None)
def romanized_hindi_sentences():
    return [[(w if PUNCT_RE.match(w) else transliterate_word(w), t) for w, t in s]
            for s in hindi_sentences("universal")]


@lru_cache(maxsize=None)
def hinglish_sentences():
    path = ROOT / "data" / "pos" / "hinglish_gold.txt"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sent = []
        for tok in line.split():
            w, _, t = tok.rpartition("/")
            sent.append((w.lower(), t))
        out.append(sent)
    return out


def splits(lang):
    if lang == "en":
        return _split(english_sentences())
    if lang == "hi":
        return _split(hindi_sentences("universal"))
    if lang == "hi-iiit":
        return _split(hindi_sentences("iiit"))
    if lang == "hg":
        return _split(hinglish_sentences())
    raise KeyError(lang)


# Training mixes for the Hinglish experiment. "hg*5" repeats the Hinglish sentences 5 times so the
# small hand-tagged set is not swamped by the larger corpora.
HINGLISH_MIXES = {
    "hg": "Hinglish only",
    "en": "English only",
    "hi-rom": "Romanized Hindi only",
    "en+hi-rom": "English + Romanized Hindi",
    "hg+en": "Hinglish + English",
    "hg+hi-rom": "Hinglish + Romanized Hindi",
    "hg+en+hi-rom": "Hinglish + English + Romanized Hindi",
    "hg*5+hi-rom": "Hinglish ×5 + Romanized Hindi",
    "hg*5+en+hi-rom": "Hinglish ×5 + English + Romanized Hindi",
}
BEST_HINGLISH_MIX = "hg*5+hi-rom"   # chosen by 5-fold cross-validation (see run_evaluation)


def hinglish_training(mix, hg_sents=None):
    parts = {
        "hg": hg_sents if hg_sents is not None else splits("hg")[0],
        "en": splits("en")[0],
        "hi-rom": _split(romanized_hindi_sentences())[0],
    }
    data = []
    for p in mix.split("+"):
        name, _, times = p.partition("*")
        data += parts[name] * int(times or 1)
    return data


def cross_validate(mix, folds=5, seed=5, factory=None):
    """k-fold cross-validated accuracy on the Hinglish sentences (more reliable than one small test split)."""
    sents = list(hinglish_sentences())
    random.Random(seed).shuffle(sents)
    parts = [sents[i::folds] for i in range(folds)]
    correct = total = 0
    for i in range(folds):
        train_hg = [s for j, f in enumerate(parts) if j != i for s in f]
        tagger = (factory or HMMTagger)().fit(hinglish_training(mix, train_hg))
        r = evaluate(tagger, parts[i])
        correct += r["accuracy"] * r["tokens"]
        total += r["tokens"]
    return correct / total


@lru_cache(maxsize=None)
def get_tagger(lang):
    """Trained tagger used by the website and the Text-to-SQL pipeline."""
    if lang == "hg":
        return HMMTagger().fit(hinglish_training(BEST_HINGLISH_MIX))
    if lang.startswith("hg:"):
        return HMMTagger().fit(hinglish_training(lang[3:]))
    return HMMTagger().fit(splits(lang)[0])


# --------------------------------------------------------------------------- evaluation
def evaluate(tagger, test_sents, known_vocab=None):
    known_vocab = known_vocab if known_vocab is not None else getattr(tagger, "word_count", {})
    gold_all, pred_all = [], []
    known_c = known_n = unk_c = unk_n = 0
    for sent in test_sents:
        words = [w for w, _ in sent]
        pred = [t for _, t in tagger.tag(words)]
        for (w, g), p in zip(sent, pred):
            gold_all.append(g)
            pred_all.append(p)
            if w in known_vocab:
                known_n += 1
                known_c += g == p
            else:
                unk_n += 1
                unk_c += g == p
    correct = sum(g == p for g, p in zip(gold_all, pred_all))
    tags = sorted(set(gold_all) | set(pred_all))
    confusion = Counter(zip(gold_all, pred_all))
    per_tag = {}
    for t in tags:
        tp = confusion[(t, t)]
        fp = sum(confusion[(g, t)] for g in tags if g != t)
        fn = sum(confusion[(t, p)] for p in tags if p != t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_tag[t] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return {
        "accuracy": correct / len(gold_all) if gold_all else 0.0,
        "tokens": len(gold_all),
        "known_accuracy": known_c / known_n if known_n else None,
        "unknown_accuracy": unk_c / unk_n if unk_n else None,
        "unknown_rate": unk_n / len(gold_all) if gold_all else 0.0,
        "per_tag": per_tag,
        "tags": tags,
        "confusion": [[confusion[(g, p)] for p in tags] for g in tags],
    }


class _NLTKWrapper:
    def __init__(self, train):
        from nltk.probability import LidstoneProbDist
        from nltk.tag.hmm import HiddenMarkovModelTrainer
        self.model = HiddenMarkovModelTrainer().train_supervised(
            train, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
        self.word_count = {w for s in train for w, _ in s}

    def tag(self, words):
        return self.model.tag(words)


def run_evaluation(include_nltk=True):
    """Every POS experiment in the report. Cached to models/pos_eval.json."""
    results = {"languages": {}, "hinglish_mixes": {}}
    for lang, label in [("en", "English (Treebank)"), ("hi", "Hindi (universal tags)"),
                        ("hi-iiit", "Hindi (IIIT tagset)"), ("hg", "Hinglish")]:
        train, test = splits(lang)
        hmm = get_tagger(lang)
        ours = evaluate(hmm, test)
        mft = evaluate(MostFrequentTagBaseline().fit(train if lang != "hg" else hinglish_training(BEST_HINGLISH_MIX)),
                       test, known_vocab=hmm.word_count)
        row = {"label": label, "train_sentences": len(train if lang != "hg" else hinglish_training(BEST_HINGLISH_MIX)),
               "test_sentences": len(test), "hmm": ours,
               "baselines": {"Most frequent tag": mft["accuracy"]}}
        if include_nltk:
            nl = _NLTKWrapper(train if lang != "hg" else hinglish_training(BEST_HINGLISH_MIX))
            row["baselines"]["NLTK HMM (Lidstone)"] = evaluate(nl, test, known_vocab=nl.word_count)["accuracy"]
        if lang == "hg":
            row["cv"] = {"HMM (ours)": cross_validate(BEST_HINGLISH_MIX),
                         "Most frequent tag": cross_validate(BEST_HINGLISH_MIX, factory=MostFrequentTagBaseline)}
        results["languages"][lang] = row
    hg_test = splits("hg")[1]
    for mix, label in HINGLISH_MIXES.items():
        tagger = get_tagger(f"hg:{mix}")
        r = evaluate(tagger, hg_test)
        results["hinglish_mixes"][mix] = {"label": label, "accuracy": r["accuracy"], "cv_accuracy": cross_validate(mix),
                                          "unknown_rate": r["unknown_rate"], "train_sentences": len(hinglish_training(mix))}
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "pos_eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def load_evaluation():
    path = MODELS_DIR / "pos_eval.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return run_evaluation()


if __name__ == "__main__":
    import sys
    import time
    sys.stdout.reconfigure(encoding="utf-8")
    t = time.time()
    res = run_evaluation()
    for lang, r in res["languages"].items():
        h = r["hmm"]
        print(f"{r['label']:28s} HMM {h['accuracy']:.3f} (known {h['known_accuracy']:.3f}, unknown "
              f"{(h['unknown_accuracy'] or 0):.3f}, unk rate {h['unknown_rate']:.2f})  baselines {r['baselines']}")
    for mix, r in res["hinglish_mixes"].items():
        print(f"  Hinglish trained on {r['label']:42s} test {r['accuracy']:.3f}  5-fold CV {r['cv_accuracy']:.3f}")
    print(f"{time.time() - t:.1f}s")
