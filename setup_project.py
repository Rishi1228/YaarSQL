"""One-time setup: rebuilds the database, dataset, trained-model caches and evaluation results.

Run:  .venv\\Scripts\\python setup_project.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, fn):
    t = time.time()
    print(f"→ {title} ...", flush=True)
    fn()
    print(f"  done in {time.time() - t:.1f}s", flush=True)


def download_corpora():
    import nltk
    target = ROOT / "data" / "nltk_data"
    for c in ["treebank", "indian", "universal_tagset", "brown"]:
        nltk.download(c, download_dir=str(target), quiet=True)


def run(*args):
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def perplexity_caches():
    sys.path.insert(0, str(ROOT))
    from nlp import ngram_service
    from nlp.corpora import CORPORA
    for f in (ROOT / "models").glob("perplexity_*.json"):
        f.unlink()
    for name in CORPORA:
        ngram_service.perplexity_table(name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    step("Downloading NLTK corpora", download_corpora)
    step("Building the college database", lambda: run("data/build_db.py"))
    step("Building the trilingual dataset", lambda: run("data/build_dataset.py"))
    step("Training + evaluating HMM POS taggers", lambda: run("-m", "nlp.hmm_pos"))
    step("Computing n-gram perplexity tables", perplexity_caches)
    step("Evaluating the Text-to-SQL pipeline", lambda: run("-m", "nlp.evaluate"))
    print("\nAll set. Start the site with:  .venv\\Scripts\\python app.py  →  http://127.0.0.1:5000")
