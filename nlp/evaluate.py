"""Phase 5 — Evaluation of the Text-to-SQL pipeline.

Metrics (as in the PARSQL paper and the proposal):
  execution accuracy  the predicted query returns the same result as the gold query
                      (row order ignored unless the gold query has ORDER BY)
  exact match         the predicted SQL string equals the gold SQL (after whitespace normalization)
  component match     per clause: table, aggregation, conditions, grouping, ordering, limit

Error categories (like Reward-SQL's breakdown): wrong table, wrong aggregation / question type,
wrong column, wrong condition, wrong grouping, wrong order / limit, no SQL.

Ablations: switch off one part of the pipeline at a time (see text2sql.Options).

Run: python -m nlp.evaluate          (writes models/eval_results.json)
"""

import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from nlp.corpora import dataset
from nlp.sql_parser import components, normalize
from nlp.text2sql import DB_PATH, Options, run
from nlp.transformer_intent import available as transformer_available

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS = MODELS_DIR / "eval_results.json"

ABLATIONS = {
    "full": ("Full system", Options()),
    "no_reasoning": ("Without reasoning-based selection", Options(use_reasoning=False)),
    "top1_intent": ("Only the top intent (no candidate selection)", Options(top_k=1)),
    "no_normalization": (
        "Without Hinglish spelling normalization",
        Options(normalize_spelling=False),
    ),
    "no_pos": ("Without POS tags", Options(use_pos=False)),
    "unigram_intent": ("Unigram intent models (Naive Bayes)", Options(intent_n=1)),
    "trigram_intent": ("Trigram intent models", Options(intent_n=3)),
}

if transformer_available():
    ABLATIONS["transformer_intent"] = (
        "Transformer intent model",
        Options(intent_model="transformer"),
    )


def _exec(con, sql):
    try:
        return con.execute(sql).fetchall(), None
    except sqlite3.Error as e:
        return None, str(e)


def same_result(con, pred, gold):
    g, _ = _exec(con, gold)
    p, err = _exec(con, pred)
    if err or p is None:
        return False
    if "ORDER BY" in gold.upper():
        return [tuple(r) for r in p] == [tuple(r) for r in g]

    def norm(rows):
        return sorted(
            tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows
        )

    return norm(p) == norm(g)


def error_category(pred, gold):
    if not pred:
        return "no SQL produced"
    try:
        p, g = components(pred), components(gold)
    except ValueError:
        return "unparseable SQL"
    if p["table"] != g["table"]:
        return "wrong table"
    if (
        p["aggregation"] != g["aggregation"]
        or (p["order"] is None) != (g["order"] is None)
        or (p["group_by"] is None) != (g["group_by"] is None)
    ):
        return "wrong question type (aggregation)"
    if p["columns"] != g["columns"] or (
        p["order"] and g["order"] and p["order"][0] != g["order"][0]
    ):
        return "wrong column"
    if p["conditions"] != g["conditions"]:
        return "wrong condition"
    if p["order"] != g["order"] or p["limit"] != g["limit"]:
        return "wrong order / limit"
    return "other"


def component_scores(pred, gold):
    keys = ["table", "aggregation", "conditions", "group_by", "order", "limit"]
    if not pred:
        return {k: 0 for k in keys}
    try:
        p, g = components(pred), components(gold)
    except ValueError:
        return {k: 0 for k in keys}
    return {k: int(p[k] == g[k]) for k in keys}


def evaluate_split(split="test", options=None, keep_examples=True):
    items = dataset(split)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    by_lang = defaultdict(Counter)
    comp = defaultdict(Counter)
    errors = defaultdict(Counter)
    examples = []
    t0 = time.time()
    for it in items:
        trace = run(it["question"], options=options, execute_sql=False)
        pred = trace["sql"]
        ex = bool(pred) and same_result(con, pred, it["sql"])
        em = bool(pred) and normalize(pred) == normalize(it["sql"])
        lang = it["lang"]
        for key in (lang, "all"):
            by_lang[key]["n"] += 1
            by_lang[key]["exec"] += ex
            by_lang[key]["exact"] += em
            by_lang[key]["lang_ok"] += trace["lang"] == lang
            for k, v in component_scores(pred, it["sql"]).items():
                comp[key][k] += v
        if not ex:
            errors[lang][error_category(pred, it["sql"])] += 1
            errors["all"][error_category(pred, it["sql"])] += 1
        if keep_examples:
            examples.append(
                {
                    "id": it["id"],
                    "lang": lang,
                    "intent": it.get("intent"),
                    "question": it["question"],
                    "gold": it["sql"],
                    "pred": pred,
                    "exec": ex,
                    "exact": em,
                }
            )
    con.close()
    summary = {}
    for key, c in by_lang.items():
        n = c["n"]
        summary[key] = {
            "n": n,
            "execution_accuracy": c["exec"] / n,
            "exact_match": c["exact"] / n,
            "language_id": c["lang_ok"] / n,
            "components": {k: v / n for k, v in comp[key].items()},
        }
    return {
        "summary": summary,
        "errors": {k: dict(v) for k, v in errors.items()},
        "examples": examples if keep_examples else [],
        "seconds": time.time() - t0,
    }


def run_all(splits=("test", "wild"), ablation_split="test"):
    MODELS_DIR.mkdir(exist_ok=True)
    out = {"splits": {}, "ablations": {}}
    for split in splits:
        if dataset(split):
            out["splits"][split] = evaluate_split(split)
    for key, (label, opt) in ABLATIONS.items():
        if key == "full":
            res = out["splits"][ablation_split]
        else:
            res = evaluate_split(ablation_split, opt, keep_examples=False)
        out["ablations"][key] = {
            "label": label,
            **{
                lang: {
                    "execution_accuracy": v["execution_accuracy"],
                    "exact_match": v["exact_match"],
                }
                for lang, v in res["summary"].items()
            },
        }
    if dataset("wild"):
        for key in ("no_reasoning", "top1_intent"):
            res = evaluate_split("wild", ABLATIONS[key][1], keep_examples=False)
            out["ablations"][key]["wild"] = res["summary"]["all"]["execution_accuracy"]
        out["ablations"]["full"]["wild"] = out["splits"]["wild"]["summary"]["all"][
            "execution_accuracy"
        ]
    from nlp.intent import evaluate as eval_intent

    out["intent"] = {str(n): eval_intent(n)[0] for n in (1, 2, 3)}
    if transformer_available():
        from nlp.transformer_intent import evaluate as eval_transformer

        out["intent"]["transformer"] = eval_transformer()
    out["validation"] = validation_scores()
    out["generated"] = time.strftime("%Y-%m-%d %H:%M")
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def validation_scores():
    """Accuracy on the validation phrasing (#3 of the train split, held out from the intent models)."""
    items = [it for it in dataset("train") if it["template"].endswith("/3")]
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    out = {}
    configs = [
        ("full", Options(holdout_template=3)),
        ("no_reasoning", Options(holdout_template=3, use_reasoning=False)),
        ("unigram_intent", Options(holdout_template=3, intent_n=1)),
        ("trigram_intent", Options(holdout_template=3, intent_n=3)),
    ]
    if transformer_available():
        configs.append(
            (
                "transformer_intent",
                Options(holdout_template=3, intent_model="transformer"),
            )
        )
    for key, opt in configs:
        ok = sum(
            same_result(
                con,
                run(it["question"], options=opt, execute_sql=False)["sql"]
                or "SELECT 0",
                it["sql"],
            )
            for it in items
        )
        out[key] = ok / len(items)
    con.close()
    return out


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    return None


def tune(split="train", sample=None):
    """Grid search for the selection weights on the TRAINING split (never on test)."""
    import random

    items = dataset(split)
    if sample:
        items = random.Random(0).sample(items, sample)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    results = []
    for wi in (0.1, 0.3, 0.6, 1.0):
        for ws in (0.0, 1.0, 2.0, 3.0, 5.0):
            for wp in (0.5, 1.0, 2.0):
                opt = Options(w_intent=wi, w_sim=ws, w_penalty=wp)
                ok = sum(
                    same_result(
                        con,
                        run(it["question"], options=opt, execute_sql=False)["sql"]
                        or "SELECT 0",
                        it["sql"],
                    )
                    for it in items
                )
                results.append((ok / len(items), wi, ws, wp))
    con.close()
    return sorted(results, reverse=True)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    res = run_all()
    for split, r in res["splits"].items():
        print(
            split,
            {
                k: (round(v["execution_accuracy"], 3), round(v["exact_match"], 3))
                for k, v in r["summary"].items()
            },
        )
        print("  errors:", r["errors"].get("all"))
    for k, a in res["ablations"].items():
        print(
            f"{a['label']:48s}",
            {
                lang: round(v["execution_accuracy"], 3)
                for lang, v in a.items()
                if isinstance(v, dict)
            },
            "wild:",
            a.get("wild"),
        )
