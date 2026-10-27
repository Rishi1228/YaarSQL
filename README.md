# PARSQL-Hinglish Lite

Ask a college database questions in **English, हिंदी or Hinglish** and get SQL + the answer.
A lab-scale version of *PARSQL: Enhancing Text-to-SQL through SQL Parsing and Reasoning*
(Dai et al., ACL Findings 2025), extended to Hindi and code-mixed Hinglish with classical NLP:
preprocessing, N-gram language models and HMM part-of-speech tagging. See `PLAN.md` for the plan
and `NLP_EXP_1.pdf` for the problem statement.

```
"computer dept me kitney studnts h?"
   → SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id
     WHERE departments.name = 'Computer'
   → 24
```

## Run it

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python setup_project.py      # corpora, database, dataset, models, evaluation (~1-2 min)
.venv\Scripts\python app.py                # open http://127.0.0.1:5000
.venv\Scripts\python -m pytest tests       # 60 tests
```

## Pages

| Page | What it shows | Coursework |
|---|---|---|
| Ask the Database | Question → SQL → answer, with every pipeline step | everything together |
| Preprocessing Lab | Normalization, tokenization, language tags, Hinglish spelling repair, stopwords, stemming | preprocessing |
| N-gram Lab | Unigram–4-gram models, 6 smoothing methods, bigram table, perplexity, generation, autocomplete, language ID, intent detection | bigrams, trigrams, n-grams |
| HMM POS Tagger | Viterbi tagging, trellis, transition/emission probabilities, accuracy for English, Hindi, Hinglish | HMM POS tagging (English + regional language) |
| Evaluation | Accuracy per language, clause-level errors, ablations, every test question | — |

## Results

| | English | Hindi | Hinglish | All |
|---|---|---|---|---|
| Text-to-SQL, test set (420 Qs, unseen phrasing) | 78.7% | 88.6% | 94.4% | **87.4%** |
| Text-to-SQL, hand-written set (60 Qs) | 85.0% | 100% | 95.0% | **93.3%** |
| HMM POS tagging | 95.1% (Treebank) | 92.3% (universal) · 83.7% (IIIT tags) | 91.7% (5-fold CV) | |

* Our HMM beats NLTK's HMM and the most-frequent-tag baseline on English and Hindi (and on Hinglish under cross-validation).
* For Hinglish, adding **Romanized Hindi** (the Hindi corpus passed through our transliterator) helps more than adding English.
* Removing the PARSQL-style reasoning-based selection drops test accuracy from 87.4% to 75.5%;
  using only the top-ranked question type drops it to 63.8%.
* All settings were tuned on a validation phrasing held out from training, never on the test set.
  A unigram intent model scores higher on test (91.7%) but lower on validation, so the bigram default was kept.

## Project layout

```
app.py                  Flask site
setup_project.py        rebuilds everything
nlp/preprocess.py       Phase 1  preprocessing
nlp/ngrams.py           Phase 2  n-gram models + smoothing (from scratch)
nlp/langid.py           Phase 2  character n-gram language identification
nlp/intent.py           Phase 2  n-gram question-type classifier
nlp/hmm_pos.py          Phase 3  HMM tagger, Viterbi, forward algorithm, evaluation
nlp/translit.py         Phase 3  Devanagari → Roman transliteration
nlp/schema.py           Phase 4  schema linking
nlp/sql_parser.py       Phase 4  PARSer-lite
nlp/reasoning.py        Phase 4  text-to-reason explanations (en / hi / hg)
nlp/text2sql.py         Phase 4  full pipeline + candidate selection
nlp/evaluate.py         Phase 5  metrics, error analysis, ablations
data/college.db         sample database (built by data/build_db.py)
data/dataset/           train / test / wild question–SQL pairs (data/build_dataset.py)
data/lexicon/           bilingual schema dictionary, Hinglish spelling variants, word lists
data/pos/               hand-tagged Hinglish POS corpus
models/                 cached evaluation results
tests/                  pytest suite
```

## Limitations

* The database and the question types are fixed (16 types over one college database); questions outside them
  (e.g. about enrollments or marks) are not supported.
* The training questions come from templates. The test set uses held-out phrasings and the hand-written set
  uses free phrasing, but real users will still find wordings the system has never seen
  ("which teacher *earns* the most" fails).
* The Hinglish POS corpus is small (104 hand-tagged sentences), so its test split is noisy; the 5-fold
  cross-validation figure is more reliable.
