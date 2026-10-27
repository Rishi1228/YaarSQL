"""Phase 2b — Intent detection with a small Transformer encoder (alternative to the n-gram models).

The question is preprocessed and masked exactly like for the n-gram classifier (<CITY>, <DEPT>,
<NUM>, ...), then fed to a Transformer encoder trained from scratch on the training questions:

    tokens → word embedding + position embedding
           → N × [multi-head self-attention → feed-forward]   (each with residual + LayerNorm)
           → mean pooling over the real (non-padding) tokens
           → linear layer → softmax over the intents

Unlike an n-gram model, self-attention lets every word look at every other word, so long-distance
cues ("kitne ... hain", "sabse ... zyada") are combined directly.

PyTorch is optional: the rest of the project runs without it. Install with  pip install torch
"""

import math
import random
from functools import lru_cache
from pathlib import Path

from nlp.corpora import question_tokens
from nlp.schema import mask

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the environment
    torch = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PAD, UNK = "<pad>", "<unk>"


def available():
    return torch is not None


if torch is not None:

    class IntentTransformer(nn.Module):
        def __init__(
            self,
            vocab_size,
            n_intents,
            d_model=64,
            heads=4,
            layers=2,
            ff=128,
            max_len=48,
            dropout=0.1,
        ):
            super().__init__()
            self.max_len = max_len
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.position = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, ff, dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                layer, layers, enable_nested_tensor=False
            )
            self.out = nn.Linear(d_model, n_intents)

        def forward(self, ids):
            pad = ids.eq(0)
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            h = self.embed(ids) * math.sqrt(self.embed.embedding_dim) + self.position(
                pos
            )
            h = self.encoder(h, src_key_padding_mask=pad)
            keep = (~pad).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
            return self.out(pooled)


class TransformerIntentClassifier:
    def __init__(
        self, epochs=40, lr=2e-3, batch_size=32, word_dropout=0.1, seed=0, **model_args
    ):
        if torch is None:
            raise ImportError("PyTorch is not installed: pip install torch")
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.word_dropout, self.seed, self.model_args = word_dropout, seed, model_args

    def features(self, tokens, pos=None, mentions=None):
        return mask(tokens, pos, mentions)

    def encode(self, words):
        ids = [self.vocab.get(w, 1) for w in words][: self.model.max_len]
        return ids or [1]

    def batch(self, seqs):
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [0] * (width - len(s)) for s in seqs])

    def fit(self, examples):
        """examples: list of (tokens, intent)"""
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        feats = [(self.features(toks), intent) for toks, intent in examples]
        self.intents = sorted({i for _, i in feats})
        words = sorted({w for f, _ in feats for w in f})
        self.vocab = {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(words)}}
        self.model = IntentTransformer(
            len(self.vocab), len(self.intents), **self.model_args
        )
        data = [(self.encode(f), self.intents.index(i)) for f, i in feats]

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * math.ceil(len(data) / self.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.model.train()
        for _ in range(self.epochs):
            rng.shuffle(data)
            for b in range(0, len(data), self.batch_size):
                chunk = data[b : b + self.batch_size]
                # word dropout: replace random words with <unk> so the model copes with unseen words
                seqs = [
                    [1 if rng.random() < self.word_dropout else t for t in ids]
                    for ids, _ in chunk
                ]
                labels = torch.tensor([y for _, y in chunk])
                loss = loss_fn(self.model(self.batch(seqs)), labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.model.eval()
        return self

    def rank(self, tokens, pos=None, mentions=None):
        """All intents sorted by probability (same format as NGramIntentClassifier.rank)."""
        with torch.no_grad():
            logits = self.model(
                self.batch([self.encode(self.features(tokens, pos, mentions))])
            )[0]
        logp = torch.log_softmax(logits, -1)
        ranked = sorted(zip(self.intents, logp.tolist()), key=lambda kv: -kv[1])
        return [{"intent": i, "logp": s, "prob": math.exp(s)} for i, s in ranked]

    def predict(self, tokens):
        return self.rank(tokens)[0]["intent"]

    def save(self, path):
        torch.save(
            {
                "vocab": self.vocab,
                "intents": self.intents,
                "model_args": self.model_args,
                "state": self.model.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.vocab, self.intents, self.model_args = (
            ckpt["vocab"],
            ckpt["intents"],
            ckpt["model_args"],
        )
        self.model = IntentTransformer(
            len(self.vocab), len(self.intents), **self.model_args
        )
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        return self


@lru_cache(maxsize=None)
def get_classifier(holdout_template=None, retrain=False):
    """Trained once, then cached in models/ so the site starts quickly."""
    name = (
        "transformer_intent.pt"
        if holdout_template is None
        else f"transformer_intent_ho{holdout_template}.pt"
    )
    path = MODELS_DIR / name
    clf = TransformerIntentClassifier()
    if path.exists() and not retrain:
        return clf.load(path)
    examples = [
        (toks, it["intent"])
        for it, toks in question_tokens("train")
        if holdout_template is None
        or not it["template"].endswith(f"/{holdout_template}")
    ]
    clf.fit(examples)
    MODELS_DIR.mkdir(exist_ok=True)
    clf.save(path)
    return clf


def evaluate(split="test"):
    """Top-1 and top-3 accuracy per language (same format as nlp.intent.evaluate)."""
    from collections import Counter

    clf = get_classifier()
    top1, top3, totals = Counter(), Counter(), Counter()
    for it, toks in question_tokens(split):
        ranked = [r["intent"] for r in clf.rank(toks)]
        totals[it["lang"]] += 1
        top1[it["lang"]] += ranked[0] == it["intent"]
        top3[it["lang"]] += it["intent"] in ranked[:3]
    out = {
        lang: {"top1": top1[lang] / totals[lang], "top3": top3[lang] / totals[lang]}
        for lang in totals
    }
    n_all = max(1, sum(totals.values()))
    out["all"] = {
        "top1": sum(top1.values()) / n_all,
        "top3": sum(top3.values()) / n_all,
    }
    return out


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    get_classifier.cache_clear()
    get_classifier(retrain=True)
    from nlp.intent import evaluate as eval_ngram

    print("bigram n-gram :", json.dumps(eval_ngram(2)[0]))
    print("transformer   :", json.dumps(evaluate()))
