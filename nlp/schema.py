"""Schema linking: find mentions of tables, columns and cell values in a tokenized question.

Uses the bilingual data dictionary in data/lexicon/schema.json, so "chhatra", "छात्रों" and
"students" all link to the `students` table.

Matching (greedy, longest span first, up to 3 tokens):
  1. exact match on the normalized surface form
  2. fuzzy match (edit distance 1) for single tokens of length >= 5 — only for tokens the
     POS tagger marked as a noun / adjective / unknown, if POS tags are given
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from nlp.preprocess import edit_distance, normalize_unicode

LEX = Path(__file__).resolve().parent.parent / "data" / "lexicon" / "schema.json"
YEAR_WORDS = {"year", "years", "yr", "saal", "varsh", "वर्ष", "साल"}
DEPT_WORDS = {"department", "departments", "dept", "vibhag", "विभाग", "branch"}
FUZZY_POS = {"NOUN", "ADJ", "X", "PROPN", None}

# Extra English synonyms people use that the dataset templates don't
EXTRA = {
    ("table", "faculty"): ["faculty", "staff", "lecturers", "lecturer"],
    ("table", "students"): ["pupils", "learners"],
    ("table", "courses"): ["course"],
    ("column", "students.cgpa"): ["gpa", "pointer", "cgpa"],
    ("column", "students.fees"): ["tuition"],
    ("column", "faculty.salary"): ["salaries", "income"],
    ("column", "faculty.experience"): ["exp"],
    ("table", "departments"): ["departmentwise", "deptwise", "विभागवार", "branch", "branches"],
}


def _norm(s):
    return tuple(normalize_unicode(s).lower().split())


@lru_cache(maxsize=1)
def schema():
    return json.loads(LEX.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def surface_index():
    """{token tuple: (type, key, value)}"""
    sc, idx = schema(), {}

    def add(form, entry):
        t = _norm(form)
        if t and t not in idx:
            idx[t] = entry

    for table, langs in sc["tables"].items():
        for forms in langs.values():
            for f in forms:
                for key in ("dir", "obl", "sing"):
                    if key in f:
                        add(f[key], ("table", table, None))
    for col, langs in sc["columns"].items():
        for forms in langs.values():
            for f in forms:
                add(f["w"], ("column", col, None))
    for col, values in sc["values"].items():
        for value, langs in values.items():
            for forms in langs.values():
                for f in forms:
                    add(f, ("value", col, value))
    for (typ, key), forms in EXTRA.items():
        for f in forms:
            add(f, (typ, key, None))
    return idx


NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


def link(tokens, pos=None, fuzzy=True):
    """Return a list of mentions: {type, key, value, start, end, text, fuzzy}.

    type: table | column | value | number
    """
    idx = surface_index()
    toks = [normalize_unicode(t).lower() for t in tokens]
    mentions, i = [], 0
    while i < len(toks):
        found = None
        for span in (3, 2, 1):
            gram = tuple(toks[i:i + span])
            if len(gram) == span and gram in idx:
                found = (span, idx[gram], False)
                break
        if not found and NUM_RE.match(toks[i]):
            v = float(toks[i])
            mentions.append({"type": "number", "key": None, "value": int(v) if v.is_integer() else v,
                             "start": i, "end": i + 1, "text": tokens[i], "fuzzy": False})
            i += 1
            continue
        if not found and fuzzy and len(toks[i]) >= 5 and (pos is None or pos[i] in FUZZY_POS):
            best, dist = None, 2
            for gram, entry in idx.items():
                if len(gram) == 1 and len(gram[0]) >= 5:
                    d = edit_distance(toks[i], gram[0])
                    if d < dist:
                        best, dist = entry, d
            if best:
                found = (1, best, True)
        if found:
            span, (typ, key, value), fz = found
            prev_tok = toks[i - 1] if i > 0 else ""
            next_tok = toks[i + span] if i + span < len(toks) else ""
            ok = True
            if typ == "value" and key == "students.year":
                # "first" is only a year in "first year" / "pehle saal" / "1st yr"
                ok = next_tok in YEAR_WORDS
            if typ == "value" and value == "IT" and toks[i] == "it":
                # English pronoun "it" is only the IT department next to a department word
                # (Hinglish/Hindi: "it mein", "it ke" are fine too)
                ok = next_tok in DEPT_WORDS or next_tok in {"mein", "me", "ke", "ka", "ki", "se", "में", "के"}
            if ok:
                mentions.append({"type": typ, "key": key, "value": value, "start": i, "end": i + span,
                                 "text": " ".join(tokens[i:i + span]), "fuzzy": fz})
                i += span
                continue
        i += 1
    return mentions


def mask(tokens, pos=None, mentions=None):
    """Replace schema mentions with placeholders: <DEPT>, <CITY>, <YEAR>, <NUM>, <TABLE>, <COL>.

    Used by the intent classifier so it learns question *shapes* ("how many <TABLE> are from
    <CITY>"), not particular tables or values. The word "department" itself is kept because it
    separates "each department" (GROUP BY) from "computer department" (WHERE).
    """
    out = list(tokens)
    names = {"departments.name": "<DEPT>", "students.city": "<CITY>", "students.year": "<YEAR>"}
    if mentions is None:
        mentions = link(tokens, pos, fuzzy=False)
    for m in sorted(mentions, key=lambda m: -m["start"]):
        if m["type"] == "value":
            tag = names[m["key"]]
        elif m["type"] == "number":
            tag = "<NUM>"
        elif m["type"] == "table" and m["key"] != "departments":
            tag = "<TABLE>"
        elif m["type"] == "column":
            tag = "<COL>"
        else:
            continue
        out[m["start"]:m["end"]] = [tag]
    return out
