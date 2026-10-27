"""Phase 4 — The Text-to-SQL pipeline (PARSQL-lite).

question
  │ 1. language ID (char n-grams)  → en / hi / hg
  │ 2. preprocessing (Phase 1)     → normalized tokens
  │ 3. HMM POS tagging (Phase 3)   → NOUN / NUM / ADJ ...
  │ 4. schema linking              → tables, columns, values, numbers
  │ 5. intent ranking (Phase 2)    → top-k question types with probabilities
  │ 6. candidate SQL for each intent (slot filling + extra filters)
  │ 7. PARSer-lite + text-to-reason → explanation of every candidate in the question's language
  │ 8. selection: intent probability + question↔reasoning similarity − penalties
  ▼ 9. execute the chosen SQL on SQLite
"""

import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nlp.hmm_pos import get_tagger
from nlp.intent import get_classifier
from nlp.langid import get_identifier
from nlp.preprocess import STOPWORDS, normalize_unicode, preprocess
from nlp.reasoning import explain
from nlp.schema import link
from nlp.sql_parser import fragments

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "college.db"

OP_OF = {
    "COUNT_ALL": "COUNT",
    "COUNT_DEPT": "COUNT",
    "COUNT_CITY": "COUNT",
    "LIST_ALL": "LIST",
    "LIST_CITY": "LIST",
    "LIST_DEPT": "LIST",
    "LIST_YEAR": "LIST",
    "LIST_GT": "LIST",
    "LIST_LT": "LIST",
    "MAX_VAL": "MAX",
    "MIN_VAL": "MIN",
    "AVG_VAL": "AVG",
    "SUM_VAL": "SUM",
    "ARGMAX": "ARGMAX",
    "ARGMIN": "ARGMIN",
    "TOP_K": "TOP_K",
    "GROUP_COUNT": "GROUP_COUNT",
    "GROUP_AVG": "GROUP_AVG",
}
NEEDS_COLUMN = {"MAX", "MIN", "AVG", "SUM", "ARGMAX", "ARGMIN", "TOP_K", "GROUP_AVG"}
GROUP_OPS = {"GROUP_COUNT", "GROUP_AVG"}
DEFAULT_COLUMN = {"students": "cgpa", "faculty": "salary", "courses": "credits"}
SUMMABLE = {"fees", "salary", "credits"}
NUMERIC_COLS = {
    "students": {"cgpa", "age", "fees", "year"},
    "faculty": {"salary", "experience"},
    "courses": {"credits"},
}

GT_WORDS = {
    normalize_unicode(w)
    for w in "greater more above over exceeds exceed exceeding higher larger bigger than zyada adhik upar jyada "
    "अधिक ज़्यादा ज्यादा ऊपर".split()
}
LT_WORDS = {
    normalize_unicode(w)
    for w in "less lower below under smaller fewer kam neeche कम नीचे".split()
}
SUPERLATIVE = {normalize_unicode(w) for w in "sabse सबसे most".split()}


@dataclass
class Options:
    """Switches used by the ablation study (Phase 5)."""

    use_reasoning: bool = True  # similarity between question and candidate explanation
    use_pos: bool = True  # POS tags guide fuzzy schema linking
    normalize_spelling: bool = True  # Hinglish spelling normalization (Phase 1)
    intent_n: int = 2  # n-gram order of the intent models
    intent_model: str = "ngram"  # "ngram" | "transformer" (Phase 2b, needs PyTorch)
    top_k: int = 16  # how many intents become SQL candidates (16 = all)
    holdout_template: int = (
        None  # validation mode: intent models trained without this phrasing
    )
    # weights chosen by grid search on a validation phrasing of the TRAIN split (see evaluate.tune)
    w_intent: float = 1.0
    w_sim: float = 6.0
    w_penalty: float = 2.0


@dataclass
class Candidate:
    intent: str
    op: str
    sql: str = None
    table: str = None
    intent_prob: float = 0.0
    penalties: list = field(default_factory=list)
    invalid: str = None
    reasoning: list = field(default_factory=list)
    fragments: list = field(default_factory=list)
    similarity: float = 0.0
    score: float = -math.inf


# --------------------------------------------------------------------------- helpers
def fmt_num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def number_roles(tokens, mentions):
    """Decide for every number whether it is a filter value (with > or <) or a count (top 5)."""
    roles = []
    for m in mentions:
        if m["type"] != "number":
            continue
        i, op, where = m["start"], None, None
        for j in range(max(0, i - 3), min(len(tokens), i + 4)):
            w = tokens[j]
            if j > 0 and tokens[j - 1] in SUPERLATIVE:
                continue
            if w in GT_WORDS and w != "than":
                op = ">"
            elif w in LT_WORDS:
                op = "<"
            if op:
                where = j
                break
        if op is None and any(tokens[j] == "than" for j in range(max(0, i - 2), i)):
            op = ">"
        roles.append({"mention": m, "op": op, "word_index": where})
    return roles


def nearest_column(index, columns):
    if not columns:
        return None
    return min(columns, key=lambda c: abs(c["start"] - index))


def build_sql(op, table, column=None, filters=(), k=None):
    """Canonical SQL (same formatting as the dataset's gold queries)."""
    join = op in GROUP_OPS or any(f["kind"] == "dept" for f in filters)
    q = (lambda c: f"{table}.{c}") if join else (lambda c: c)
    src = (
        f"{table} JOIN departments ON {table}.dept_id = departments.id"
        if join
        else table
    )
    conds = []
    order = {"dept": 0, "city": 1, "year": 2, "num": 3}
    for f in sorted(filters, key=lambda f: order[f["kind"]]):
        if f["kind"] == "dept":
            conds.append(f"departments.name = '{f['value']}'")
        elif f["kind"] == "city":
            conds.append(f"{q('city')} = '{f['value']}'")
        elif f["kind"] == "year":
            conds.append(f"{q('year')} = {f['value']}")
        else:
            conds.append(f"{q(f['column'])} {f['op']} {fmt_num(f['value'])}")
    where = f" WHERE {' AND '.join(conds)}" if conds else ""
    if op == "COUNT":
        return f"SELECT COUNT(*) FROM {src}{where}"
    if op == "LIST":
        return f"SELECT {q('name')} FROM {src}{where}"
    if op in ("MAX", "MIN", "AVG", "SUM"):
        return f"SELECT {op}({q(column)}) FROM {src}{where}"
    if op in ("ARGMAX", "ARGMIN"):
        d = "DESC" if op == "ARGMAX" else "ASC"
        return f"SELECT {q('name')} FROM {src}{where} ORDER BY {q(column)} {d} LIMIT 1"
    if op == "TOP_K":
        return (
            f"SELECT {q('name')} FROM {src}{where} ORDER BY {q(column)} DESC LIMIT {k}"
        )
    if op == "GROUP_COUNT":
        return f"SELECT departments.name, COUNT(*) FROM {src}{where} GROUP BY departments.name"
    if op == "GROUP_AVG":
        return f"SELECT departments.name, AVG({q(column)}) FROM {src}{where} GROUP BY departments.name"
    raise ValueError(op)


# --------------------------------------------------------------------------- candidates
def make_candidate(intent, prob, mentions, roles, tokens):
    op = OP_OF[intent]
    cand = Candidate(intent=intent, op=op, intent_prob=prob)
    tables = [m for m in mentions if m["type"] == "table" and m["key"] != "departments"]
    dept_word = [
        m for m in mentions if m["type"] == "table" and m["key"] == "departments"
    ]
    columns = [m for m in mentions if m["type"] == "column"]
    values = [m for m in mentions if m["type"] == "value"]
    used = set()

    # numeric filters (from comparator words); LIST_GT / LIST_LT force a direction
    num_filters, free_numbers = [], []
    for r in roles:
        op_ = r["op"]
        if op_ is None and intent in ("LIST_GT", "LIST_LT") and not num_filters:
            op_ = ">" if intent == "LIST_GT" else "<"
        if op_:
            num_filters.append((r["mention"], op_))
        else:
            free_numbers.append(r["mention"])

    # column for the aggregation / ordering
    filter_cols = {}
    for m, _ in num_filters:
        c = nearest_column(m["start"], columns)
        filter_cols[id(m)] = c
    agg_col = None
    if op in NEEDS_COLUMN:
        spare = [c for c in columns if all(c is not fc for fc in filter_cols.values())]
        agg_col = spare[0] if spare else (columns[0] if columns else None)

    # table
    if agg_col:
        table = agg_col["key"].split(".")[0]
    elif tables:
        table = tables[0]["key"]
    elif columns:
        table = columns[0]["key"].split(".")[0]
    elif any(v["key"] in ("students.city", "students.year") for v in values):
        table = "students"
    elif dept_word and op in ("COUNT", "LIST") and not values:
        table = "departments"
    else:
        table = "students"
        cand.penalties.append("no table mentioned (assumed students)")
    if tables and tables[0]["key"] != table:
        cand.penalties.append(
            f"question mentions {tables[0]['key']} but query uses {table}"
        )
    for t in tables:
        if t["key"] == table:
            used.add(id(t))
    cand.table = table

    column = None
    if op in NEEDS_COLUMN:
        if agg_col:
            column = agg_col["key"].split(".")[1]
            used.add(id(agg_col))
        elif table in DEFAULT_COLUMN:
            column = DEFAULT_COLUMN[table]
            cand.penalties.append(f"no column mentioned (assumed {column})")
        else:
            cand.invalid = "this question type needs a numeric column"
            return cand
        if op == "SUM" and column not in SUMMABLE:
            cand.invalid = f"adding up {column} makes no sense"
            return cand

    # filters
    filters = []
    for v in values:
        kind = {
            "departments.name": "dept",
            "students.city": "city",
            "students.year": "year",
        }[v["key"]]
        if kind in ("city", "year") and table != "students":
            cand.invalid = f"{table} have no {kind}"
            return cand
        if kind == "dept" and (table == "departments" or op in GROUP_OPS):
            cand.invalid = "department filter does not fit this question type"
            return cand
        filters.append(
            {"kind": kind, "value": int(v["value"]) if kind == "year" else v["value"]}
        )
        used.add(id(v))
    for m, op_ in num_filters:
        c = filter_cols[id(m)]
        if c and c["key"].split(".")[0] == table:
            col = c["key"].split(".")[1]
            used.add(id(c))
        elif table in DEFAULT_COLUMN:
            col = column or DEFAULT_COLUMN[table]
            cand.penalties.append(
                f"no column for the number {m['value']} (assumed {col})"
            )
        else:
            cand.invalid = "numeric filter on a table without numeric columns"
            return cand
        filters.append({"kind": "num", "column": col, "op": op_, "value": m["value"]})
        used.add(id(m))

    k = None
    if op == "TOP_K":
        ints = [
            m
            for m in free_numbers
            if isinstance(m["value"], int) and 0 < m["value"] <= 100
        ]
        if ints:
            k = ints[0]["value"]
            used.add(id(ints[0]))
        else:
            cand.invalid = "top-k needs a number (top 5, top teen ...)"
            return cand
    if dept_word and (
        op in GROUP_OPS
        or any(f["kind"] == "dept" for f in filters)
        or table == "departments"
    ):
        for d in dept_word:
            used.add(id(d))

    for m in mentions:
        if id(m) not in used:
            what = m["text"]
            cand.penalties.append(f"“{what}” is not used by this query")
    if op in GROUP_OPS and not dept_word:
        cand.penalties.append(
            "grouping by department but the question never says department"
        )

    if op in ("ARGMAX", "ARGMIN", "TOP_K") and any(
        f["kind"] == "num" and f["column"] == column for f in filters
    ):
        cand.penalties.append(f"filters and sorts on the same column ({column})")

    cand.sql = build_sql(op, table, column, filters, k)
    return cand


def similarity(q_tokens, r_tokens, lang):
    """How well a candidate's explanation matches the question (0..1).

    0.6 × share of the question's content words found in the explanation
    + 0.4 × cosine similarity of character trigrams.
    """
    sw = {normalize_unicode(w) for w in STOPWORDS.get(lang, set())}
    content = [w for w in q_tokens if w not in sw] or q_tokens
    rset = set(r_tokens)
    rpre = {w[:4] for w in r_tokens if len(w) >= 4}
    hit = sum(1 for w in content if w in rset or (len(w) >= 4 and w[:4] in rpre))
    recall = hit / len(content) if content else 0.0

    def grams(ts):
        s = " " + " ".join(ts) + " "
        return Counter(s[i : i + 3] for i in range(len(s) - 2))

    a, b = grams(q_tokens), grams(r_tokens)
    dot = sum(a[g] * b[g] for g in a)
    cos = dot / (
        math.sqrt(sum(v * v for v in a.values()))
        * math.sqrt(sum(v * v for v in b.values()))
        or 1
    )
    return 0.6 * recall + 0.4 * cos


def execute(sql, limit=100):
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(limit + 1)
        return {
            "columns": cols,
            "rows": [list(r) for r in rows[:limit]],
            "truncated": len(rows) > limit,
        }
    finally:
        con.close()


# --------------------------------------------------------------------------- pipeline
def run(question, lang=None, options=None, execute_sql=True):
    opt = options or Options()
    trace = {"question": question}

    lid = get_identifier()
    trace["lang_scores"] = lid.scores(question)["posterior"]
    lang = lang or lid.predict(question)
    trace["lang"] = lang

    pre = preprocess(question, lang, normalize_spelling=opt.normalize_spelling)
    tokens = pre["final"]
    trace["tokens"] = tokens
    trace["spelling_changes"] = pre["spelling_changes"]

    pos = [t for _, t in get_tagger(lang).tag(tokens)] if tokens else []
    trace["pos"] = [{"word": w, "tag": t} for w, t in zip(tokens, pos)]

    mentions = link(tokens, pos if opt.use_pos else None, fuzzy=True)
    if not opt.use_pos:
        # without POS tags we cannot tell a misspelled noun from any other word: fuzzy-match everything
        mentions = link(tokens, None, fuzzy=True)
    trace["mentions"] = [
        {k: m[k] for k in ("type", "key", "value", "text", "fuzzy", "start", "end")}
        for m in mentions
    ]

    if not mentions:
        # nothing in the question refers to the database: refuse instead of guessing
        trace.update(
            {
                "numbers": [],
                "intents": [],
                "candidates": [],
                "sql": None,
                "reasoning": [],
                "error": "The question doesn't mention anything in the database (students, teachers, "
                "courses, departments, cities, cgpa, salary ...).",
            }
        )
        return trace

    roles = number_roles(tokens, mentions)
    trace["numbers"] = [
        {
            "value": r["mention"]["value"],
            "role": f"filter {r['op']}" if r["op"] else "count / limit",
        }
        for r in roles
    ]

    if opt.intent_model == "transformer":
        from nlp.transformer_intent import get_classifier as get_transformer

        clf = get_transformer(holdout_template=opt.holdout_template)
    else:
        clf = get_classifier(opt.intent_n, holdout_template=opt.holdout_template)
    ranking = clf.rank(tokens, pos, mentions) if tokens else []
    trace["intents"] = ranking[: opt.top_k]

    # comparison words used by a numeric filter ("8 se zyada") must not also count as
    # evidence for "sabse zyada" (highest) when comparing the question with explanations
    filter_words = {r["word_index"] for r in roles if r["word_index"] is not None}
    sim_tokens = [w for i, w in enumerate(tokens) if i not in filter_words]

    candidates = []
    for r in ranking[: opt.top_k]:
        c = make_candidate(r["intent"], r["prob"], mentions, roles, tokens)
        if c.sql:
            c.reasoning = explain(c.sql, lang)
            c.fragments = fragments(c.sql)
            r_tokens = preprocess(" ".join(c.reasoning), lang)["final"]
            c.similarity = similarity(sim_tokens, r_tokens, lang)
            c.score = (
                opt.w_intent * math.log(max(c.intent_prob, 1e-4))
                + (opt.w_sim * c.similarity if opt.use_reasoning else 0.0)
                - opt.w_penalty * len(c.penalties)
            )
        candidates.append(c)

    # the same SQL can come from several intents (COUNT_ALL / COUNT_CITY ...): keep the best score
    best_by_sql = {}
    for c in candidates:
        if c.sql and (c.sql not in best_by_sql or c.score > best_by_sql[c.sql].score):
            best_by_sql[c.sql] = c
    valid = sorted(best_by_sql.values(), key=lambda c: -c.score)
    chosen = valid[0] if valid else None

    trace["candidates"] = [
        {
            "intent": c.intent,
            "op": c.op,
            "sql": c.sql,
            "intent_prob": c.intent_prob,
            "similarity": c.similarity,
            "penalties": c.penalties,
            "invalid": c.invalid,
            "score": None if c.score == -math.inf else c.score,
            "reasoning": c.reasoning,
            "fragments": c.fragments,
            "chosen": c is chosen,
        }
        for c in sorted(candidates, key=lambda c: -c.score if c.sql else math.inf)
    ]
    trace["sql"] = chosen.sql if chosen else None
    trace["reasoning"] = chosen.reasoning if chosen else []
    if chosen and execute_sql:
        try:
            trace["result"] = execute(chosen.sql)
        except sqlite3.Error as e:
            trace["result"] = {"error": str(e)}
    return trace
