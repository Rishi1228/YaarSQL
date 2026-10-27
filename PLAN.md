# Implementation Plan — PARSQL-Hinglish Lite

> **Status (2026-10-26): all phases 0–6 implemented.** See `README.md` for how to run it and the results.
> Changes from the original plan: the Hinglish POS corpus is 104 hand-tagged sentences, boosted with
> Romanized Hindi (a Devanagari→Roman transliterator was added); a 60-question hand-written "wild"
> test set was added; the About content lives on the Overview page.

A desktop website that turns **English / Hindi / Hinglish** questions into **SQL**, built with
classical NLP (preprocessing, N-gram models, HMM POS tagging). It's a lab-scale version of the
idea in `NLP_EXP_1.pdf`: PARSQL's parse → reason → select approach, applied to Hindi and Hinglish.

> Example: `computer department mein kitne students hain?`
> → `SELECT COUNT(*) FROM students s JOIN departments d ON s.dept_id = d.id WHERE d.name = 'Computer';`

---

## 1. Scope (what's realistic for a lab project)

The paper fine-tunes 3B-parameter models on Spider/BIRD. We don't. We keep the same **ideas**
and build them with the coursework techniques:

| Paper idea (PARSQL)                       | Our lab-scale version                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Natural-language question understanding   | Preprocessing + N-gram models + HMM POS tags                                                                       |
| PARSer: parse SQL into parts              | Small SQL parser that splits a query into clauses (SELECT / WHERE / GROUP BY / ORDER BY / JOIN)                    |
| Text-to-reason (explanation for each SQL) | Rule-based reasoning templates in English, Hindi and Hinglish                                                      |
| Similarity-based candidate selection      | Make top-k SQL candidates, score each one's reasoning text against the question with n-gram overlap, pick the best |
| Hinglish-Spider-Aug dataset               | Our own small trilingual dataset over one sample database                                                          |
| Execution accuracy / exact match          | Same metrics, split by language                                                                                    |

## 2. Tech stack

- **Backend:** Python 3 + Flask. Every NLP algorithm (n-grams, smoothing, HMM, Viterbi) is
  **written from scratch** so it counts as coursework. NLTK is used only for **corpora**
  (and as a baseline to compare our HMM against).
- **Database:** SQLite (`college.db`): departments, students, faculty, courses, enrollments.
- **Frontend:** Server-rendered HTML + vanilla JS + CSS, laid out for desktop (≥1280px);
  Chart.js for graphs.
- **Corpora:**
  - English POS: NLTK Penn Treebank sample (~3,900 sentences), universal tagset.
  - Hindi POS: NLTK `indian` corpus (`hindi.pos`, ~540 tagged sentences, Devanagari).
  - Hinglish POS: our own hand-tagged set (~150–200 sentences, universal tagset).
  - Text-to-SQL: our own trilingual question↔SQL dataset (~600–900 pairs, generated from
    templates + hand-written paraphrases), split into train / test.

## 3. Folder layout

```
NLP_miniProject/
├── app.py                  # Flask entry point
├── requirements.txt
├── nlp/
│   ├── preprocess.py       # Phase 1
│   ├── ngrams.py           # Phase 2
│   ├── langid.py           # Phase 2 (char n-gram language ID)
│   ├── hmm_pos.py          # Phase 3
│   ├── sql_parser.py       # Phase 4 (PARSer-lite)
│   ├── reasoning.py        # Phase 4 (text-to-reason templates)
│   ├── text2sql.py         # Phase 4 (pipeline)
│   └── evaluate.py         # Phase 5
├── data/
│   ├── college.db          # sample database
│   ├── build_db.py
│   ├── dataset/            # trilingual question-SQL pairs (train/test JSON)
│   ├── hinglish_pos.txt    # hand-tagged Hinglish sentences
│   └── lexicon/            # Hinglish spelling variants, Hindi↔English schema words, stopwords
├── models/                 # trained n-gram / HMM models (saved as JSON/pickle)
├── templates/ static/      # website
└── tests/
```

---

## 4. Phases

### Phase 0 — Setup, database, dataset

- Virtual environment, `requirements.txt` (flask, nltk), download NLTK corpora.
- Build `college.db` with realistic sample rows.
- Build the trilingual dataset: ~15 intent templates (count, list, filter, max/min, average,
  group-by, order-by/top-k, join) × slot values × 3 languages, plus hand-written paraphrases
  so it isn't purely templated. Every item: `{id, lang, question, sql, intent}`.
- **Deliverable:** working Flask skeleton with nav bar; DB + dataset files.

### Phase 1 — Preprocessing _(coursework: preprocessing)_

- Unicode normalization (NFC), lowercasing (Latin only — Devanagari has no case).
- Sentence splitting (handles `.` `?` `!` and Hindi `।`).
- Tokenization: regex tokenizer that keeps numbers, quoted values, Devanagari words, and
  English contractions intact.
- **Script detection per token:** Latin / Devanagari / digit / punctuation.
- **Hinglish spelling normalization:** maps variants to one form
  (`kitne/kitnay/ktne → kitne`, `hai/h/hain → hai`) using a lexicon + edit distance.
- Stopword removal (English + Hindi + Hinglish lists) — optional so the n-gram/HMM steps can
  keep function words.
- Stemming: Porter (English) + a simple suffix stripper for Hindi/Hinglish
  (`-on`, `-ein`, `-iyan`, `-ों`, `-ें`…).
- Number words → digits (`five`, `paanch`, `पाँच` → `5`).
- **Website page — "Preprocessing Lab":** type text, see each step side by side as a table
  (original → normalized → tokens → script tags → stopwords removed → stems).

### Phase 2 — N-grams: unigrams, bigrams, trigrams, general N _(coursework: bigrams, trigrams)_

- `NGramModel(n)` built from scratch: counts, conditional probabilities with `<s>` / `</s>` padding.
- Smoothing: MLE, Laplace (add-1), add-k, linear interpolation, and stupid backoff.
- **Perplexity** on held-out text; compare unigram vs bigram vs trigram vs 4-gram for each language.
- Text generation (sample from the model) and **next-word autocomplete**.
- Uses inside the project:
  1. **Autocomplete** in the question box (trigram → bigram → unigram backoff).
  2. **Language ID:** character trigram models for English / Hindi / Hinglish; pick the one with the
     highest log-probability.
  3. **Intent classification:** one word-bigram LM per intent (COUNT, MAX, GROUP BY…); the
     question goes to the intent whose LM gives it the highest probability (also scored against a
     Naive Bayes baseline).
  4. **Candidate scoring** for Phase 4 (n-gram overlap between question and reasoning text).
- **Website page — "N-gram Lab":** choose corpus + N + smoothing; show top n-grams with frequency
  bar chart, the probability table, perplexity comparison chart, generated sentences,
  and a live autocomplete demo.

### Phase 3 — POS tagging with HMM _(coursework: HMM POS tagging, English + regional language)_

- `HMMTagger` from scratch:
  - Transition probs P(tagᵢ | tagᵢ₋₁) from bigram tag counts (optional trigram-tag variant).
  - Emission probs P(word | tag) with smoothing.
  - Unknown words: suffix-based emission estimate + shape features (digit, capitalised, Devanagari).
  - **Viterbi decoding** (log space), plus the forward algorithm for sentence probability.
- **3a. English:** train on Treebank (universal tagset), 80/20 split, report accuracy +
  per-tag precision/recall + confusion matrix; compare with NLTK's built-in HMM.
- **3b. Hindi (regional language):** train on NLTK `hindi.pos`, same evaluation.
- **3c. Hinglish (our mini-project's language):** train on hand-tagged Hinglish set (optionally
  combined with English + Hindi mapped to a shared universal tagset) and report how each
  training mix performs.
- Uses inside the project: NOUN/PROPN tokens → candidate tables/columns/values; NUM → numeric
  values / LIMIT; ADJ like _highest / sabse zyada / सबसे अधिक_ → MAX / ORDER BY DESC.
- **Website page — "POS Tagger":** choose language, enter sentence, see colour-coded tags,
  the **Viterbi trellis** visualised as a grid, transition-matrix heatmap, and the accuracy dashboard.

### Phase 4 — Text-to-SQL pipeline (PARSQL-lite)

1. Preprocess (Phase 1) → language ID (Phase 2) → POS tags (Phase 3).
2. **Schema linking:** match tokens/n-grams to tables, columns and cell values using a
   bilingual schema lexicon (`chhatra / छात्र → students`) + fuzzy matching.
3. **Intent:** n-gram intent classifier gives top-k intents.
4. **Slot filling** → build top-k SQL candidates.
5. **PARSer-lite:** parse each candidate SQL into clauses.
6. **Text-to-reason:** rule-based explanation for each candidate, written in the question's language.
7. **Selection:** score (question ↔ reasoning) with n-gram overlap + intent probability; pick the best.
8. Execute on SQLite, show the result table.

- **Website page — "Ask the Database" (home):** question box with autocomplete, the result, and
  an expandable "How it worked" panel showing every step above.

### Phase 5 — Evaluation & ablations

- Execution accuracy + exact-match accuracy, separately for English / Hindi / Hinglish test sets.
- Clause-level errors (wrong table vs wrong aggregation vs wrong condition), like the paper's error analysis.
- Ablations: (a) without reasoning-based selection, (b) without Hinglish normalization,
  (c) bigram vs trigram intent models, (d) without POS features.
- **Website page — "Evaluation":** charts + tables for all of the above.

### Phase 6 — Polish & report

- About page tying the project back to the PARSQL paper.
- README (how to run), unit tests for each module, short write-up of results for the lab journal.

---

## 5. Order of work & checkpoints

| Phase | What you can show after it                                                    |
| ----- | ----------------------------------------------------------------------------- |
| 0     | Site runs, DB + dataset exist                                                 |
| 1     | Preprocessing Lab page works                                                  |
| 2     | N-gram Lab page: bigram/trigram tables, perplexity, autocomplete, language ID |
| 3     | POS Tagger page: HMM tagging in English, Hindi, Hinglish + accuracy           |
| 4     | Ask the Database: question → SQL → results                                    |
| 5     | Evaluation page with accuracy per language                                    |
| 6     | Final polish + README                                                         |

Each phase ends with a check-in so you can review before the next one starts.
