"""PARSQL-Hinglish Lite — Flask entry point.

Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:5000
"""
import nltk
from flask import Flask, jsonify, render_template, request

from pathlib import Path

ROOT = Path(__file__).resolve().parent
nltk.data.path.insert(0, str(ROOT / "data" / "nltk_data"))

from nlp import ngram_service, pos_service, text2sql  # noqa: E402
from nlp.hmm_pos import BEST_HINGLISH_MIX, HINGLISH_MIXES  # noqa: E402
from nlp.corpora import CORPORA  # noqa: E402
from nlp.ngrams import SMOOTHINGS  # noqa: E402
from nlp.preprocess import preprocess  # noqa: E402  (needs the nltk path above)

app = Flask(__name__)
app.json.ensure_ascii = False  # send Devanagari as-is

# Sidebar navigation.
NAV = [
    {"endpoint": "home", "label": "Overview", "icon": "◆", "phase": None, "ready": True},
    {"endpoint": "ask", "label": "Ask the Database", "icon": "⌕", "phase": 4, "ready": True},
    {"endpoint": "preprocess_page", "label": "Preprocessing Lab", "icon": "¶", "phase": 1, "ready": True},
    {"endpoint": "ngram_page", "label": "N-gram Lab", "icon": "≡", "phase": 2, "ready": True},
    {"endpoint": "pos_page", "label": "HMM POS Tagger", "icon": "⋯", "phase": 3, "ready": True},
    {"endpoint": "eval_page", "label": "Evaluation", "icon": "▤", "phase": 5, "ready": True},
]

EXAMPLES = {
    "en": "How many students are in the Computer department?",
    "hi": "कंप्यूटर विभाग में कितने छात्र हैं? पाँच से ज़्यादा।",
    "hg": "computer dept me kitney studnts h?",
    "hg2": "top paanch students ka naam btao jinka cgpa 8.5 se jyada hai",
}


@app.context_processor
def inject_nav():
    return {"nav": NAV}


def home_stats():
    from nlp.evaluate import load_results
    stats = {"test": "–", "wild": "–", "pos_en": "–", "pos_hi": "–", "pos_hg": "–"}
    res = load_results()
    if res:
        stats["test"] = f"{100 * res['splits']['test']['summary']['all']['execution_accuracy']:.1f}%"
        if "wild" in res["splits"]:
            stats["wild"] = f"{100 * res['splits']['wild']['summary']['all']['execution_accuracy']:.1f}%"
    pos_file = ROOT / "models" / "pos_eval.json"
    if pos_file.exists():
        import json
        pos = json.loads(pos_file.read_text(encoding="utf-8"))["languages"]
        stats["pos_en"] = f"{100 * pos['en']['hmm']['accuracy']:.0f}%"
        stats["pos_hi"] = f"{100 * pos['hi']['hmm']['accuracy']:.0f}%"
        stats["pos_hg"] = f"{100 * pos['hg'].get('cv', {}).get('HMM (ours)', pos['hg']['hmm']['accuracy']):.1f}%"
    return stats


@app.route("/")
def home():
    return render_template("home.html", stats=home_stats())


ASK_EXAMPLES = [
    ("Hinglish", "computer dept me kitney studnts h?"),
    ("Hinglish", "sabse zyada salary wala teacher kaun hai"),
    ("Hinglish", "pune ke students ka average cgpa"),
    ("हिंदी", "हर विभाग में कितने पाठ्यक्रम हैं"),
    ("हिंदी", "9 से ज़्यादा सीजीपीए वाले छात्रों के नाम"),
    ("English", "top 5 students based on cgpa"),
    ("English", "how many students in the IT department have cgpa above 8"),
]


def schema_overview():
    import sqlite3
    con = sqlite3.connect(f"file:{text2sql.DB_PATH}?mode=ro", uri=True)
    tables = []
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        tables.append({"name": name, "columns": cols, "rows": count})
    con.close()
    return tables


@app.route("/ask")
def ask():
    return render_template("ask.html", examples=ASK_EXAMPLES, tables=schema_overview())


@app.route("/preprocess")
def preprocess_page():
    return render_template("preprocess.html", examples=EXAMPLES)


@app.route("/ngrams")
def ngram_page():
    return render_template("ngrams.html", corpora=CORPORA, smoothings=SMOOTHINGS)


@app.route("/pos")
def pos_page():
    return render_template("pos.html", mixes=HINGLISH_MIXES, best_mix=BEST_HINGLISH_MIX)


@app.route("/evaluation")
def eval_page():
    return render_template("evaluation.html")


@app.post("/api/preprocess")
def api_preprocess():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Enter some text first."}), 400
    lang = body.get("lang") or None
    if lang not in (None, "en", "hi", "hg"):
        lang = None
    return jsonify(preprocess(text, lang=lang, drop_stopwords=bool(body.get("drop_stopwords"))))


def _text():
    body = request.get_json(force=True) or {}
    return (body.get("text") or "").strip(), body


def _ngram_args(src):
    corpus_name = src.get("corpus", "questions-hg")
    smoothing = src.get("smoothing", "kneser_ney")
    try:
        n = max(1, min(4, int(src.get("n", 2))))
    except (TypeError, ValueError):
        n = 2
    if corpus_name not in CORPORA or smoothing not in SMOOTHINGS:
        return None
    return corpus_name, n, smoothing


@app.get("/api/ngram/overview")
def api_ngram_overview():
    args = _ngram_args(request.args)
    if not args:
        return jsonify({"error": "Unknown corpus or smoothing."}), 400
    return jsonify(ngram_service.overview(*args))


@app.get("/api/ngram/perplexity")
def api_ngram_perplexity():
    name = request.args.get("corpus", "questions-hg")
    if name not in CORPORA:
        return jsonify({"error": "Unknown corpus."}), 400
    return jsonify(ngram_service.perplexity_table(name))


@app.post("/api/ngram/sentence")
def api_ngram_sentence():
    text, body = _text()
    args = _ngram_args(body)
    if not text or not args:
        return jsonify({"error": "Enter a sentence."}), 400
    return jsonify(ngram_service.sentence(*args, text))


@app.get("/api/autocomplete")
def api_autocomplete():
    lang = request.args.get("lang") or None
    return jsonify(ngram_service.autocomplete(request.args.get("text", ""), lang if lang in ("en", "hi", "hg") else None))


@app.post("/api/langid")
def api_langid():
    text, _ = _text()
    if not text:
        return jsonify({"error": "Enter some text."}), 400
    return jsonify(ngram_service.langid(text))


@app.post("/api/intent")
def api_intent():
    text, _ = _text()
    if not text:
        return jsonify({"error": "Enter a question."}), 400
    return jsonify(ngram_service.intent(text))


@app.post("/api/ask")
def api_ask():
    text, body = _text()
    if not text:
        return jsonify({"error": "Type a question first."}), 400
    if len(text) > 300:
        return jsonify({"error": "Please keep the question under 300 characters."}), 400
    lang = body.get("lang") if body.get("lang") in ("en", "hi", "hg") else None
    trace = text2sql.run(text, lang)
    if not trace.get("sql") and not trace.get("error"):
        trace["error"] = "Sorry, I couldn't turn that into a query. Try mentioning a table (students, teachers, courses)."
    return jsonify(trace)


@app.get("/api/eval")
def api_eval():
    from nlp.evaluate import load_results
    res = load_results()
    if res is None:
        return jsonify({"error": "No results yet. Run: python -m nlp.evaluate"}), 404
    pos = pos_service.evaluation()
    res["pos"] = {k: {"label": v["label"], "accuracy": v["hmm"]["accuracy"]} for k, v in pos["languages"].items()}
    return jsonify(res)


@app.post("/api/eval/run")
def api_eval_run():
    from nlp.evaluate import run_all
    run_all()
    return api_eval()


POS_LANGS = ("en", "hi", "hi-iiit", "hg")


@app.post("/api/pos/tag")
def api_pos_tag():
    text, body = _text()
    if not text:
        return jsonify({"error": "Enter a sentence."}), 400
    lang = body.get("lang") or None
    if lang not in POS_LANGS:
        lang = None
    return jsonify(pos_service.tag(text, lang, body.get("mix")))


@app.get("/api/pos/model")
def api_pos_model():
    lang = request.args.get("lang", "en")
    if lang not in POS_LANGS:
        return jsonify({"error": "Unknown language."}), 400
    return jsonify(pos_service.model_info(lang, request.args.get("mix")))


@app.get("/api/pos/eval")
def api_pos_eval():
    return jsonify(pos_service.evaluation())


if __name__ == "__main__":
    app.run(debug=True)
