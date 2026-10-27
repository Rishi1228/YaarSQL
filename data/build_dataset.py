"""Builds the trilingual Text-to-SQL dataset (our small "Hinglish-Spider-Aug").

Every item: {id, lang, intent, question, sql, template}
  lang   : en (English) | hi (Hindi, Devanagari) | hg (Hinglish, Romanized code-mixed)
  intent : one of the 16 question types below

Split: for each (intent, language) the LAST template is held out for the test set, so test
questions use phrasing that never appears in training.

Run: python data/build_dataset.py
"""
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SCHEMA = json.loads((HERE / "lexicon" / "schema.json").read_text(encoding="utf-8"))
VARIANTS = json.loads((HERE / "lexicon" / "hinglish_variants.json").read_text(encoding="utf-8"))
OUT = HERE / "dataset"

FILLS_PER_TEMPLATE = 8
NOISE_PROB = 0.35  # chance a Hinglish word is replaced by a spelling variant

# ---------------------------------------------------------------------------
# SQL for each intent. {t} table, {c} column, {v} value, {k} limit
# ---------------------------------------------------------------------------
JOIN = "{t} JOIN departments ON {t}.dept_id = departments.id"
SQL = {
    "COUNT_ALL":   "SELECT COUNT(*) FROM {t}",
    "COUNT_DEPT":  "SELECT COUNT(*) FROM " + JOIN + " WHERE departments.name = '{d}'",
    "COUNT_CITY":  "SELECT COUNT(*) FROM {t} WHERE city = '{city}'",
    "LIST_ALL":    "SELECT name FROM {t}",
    "LIST_CITY":   "SELECT name FROM {t} WHERE city = '{city}'",
    "LIST_DEPT":   "SELECT {t}.name FROM " + JOIN + " WHERE departments.name = '{d}'",
    "LIST_YEAR":   "SELECT name FROM {t} WHERE year = {y}",
    "LIST_GT":     "SELECT name FROM {t} WHERE {c} > {v}",
    "LIST_LT":     "SELECT name FROM {t} WHERE {c} < {v}",
    "MAX_VAL":     "SELECT MAX({c}) FROM {t}",
    "MIN_VAL":     "SELECT MIN({c}) FROM {t}",
    "AVG_VAL":     "SELECT AVG({c}) FROM {t}",
    "SUM_VAL":     "SELECT SUM({c}) FROM {t}",
    "ARGMAX":      "SELECT name FROM {t} ORDER BY {c} DESC LIMIT 1",
    "ARGMIN":      "SELECT name FROM {t} ORDER BY {c} ASC LIMIT 1",
    "TOP_K":       "SELECT name FROM {t} ORDER BY {c} DESC LIMIT {k}",
    "GROUP_COUNT": "SELECT departments.name, COUNT(*) FROM " + JOIN + " GROUP BY departments.name",
    "GROUP_AVG":   "SELECT departments.name, AVG({t}.{c}) FROM " + JOIN + " GROUP BY departments.name",
}

# Which tables / columns each intent may use
ALL_T = ["students", "faculty", "courses", "departments"]
DEPT_T = ["students", "faculty", "courses"]
STU = ["students"]
NUM_COLS = list(SCHEMA["columns"])                      # e.g. "students.cgpa"
SUM_COLS = ["students.fees", "faculty.salary", "courses.credits"]
NAMED_COLS = NUM_COLS                                    # tables with a name column

SLOTS = {
    "COUNT_ALL": {"t": ALL_T}, "LIST_ALL": {"t": ALL_T},
    "COUNT_DEPT": {"t": DEPT_T, "d": True}, "LIST_DEPT": {"t": DEPT_T, "d": True},
    "COUNT_CITY": {"t": STU, "city": True}, "LIST_CITY": {"t": STU, "city": True},
    "LIST_YEAR": {"t": STU, "y": True},
    "LIST_GT": {"c": NUM_COLS, "v": True}, "LIST_LT": {"c": NUM_COLS, "v": True},
    "MAX_VAL": {"c": NUM_COLS}, "MIN_VAL": {"c": NUM_COLS}, "AVG_VAL": {"c": NUM_COLS},
    "SUM_VAL": {"c": SUM_COLS},
    "ARGMAX": {"c": NAMED_COLS}, "ARGMIN": {"c": NAMED_COLS}, "TOP_K": {"c": NAMED_COLS, "k": True},
    "GROUP_COUNT": {"t": DEPT_T}, "GROUP_AVG": {"c": NUM_COLS},
}

# ---------------------------------------------------------------------------
# Templates. Slots: {T} plural table, {T_obl} Hindi oblique plural, {T_s} singular,
# {C} column, {ka} ka/ki/का/की agreeing with the column, {D} department, {CITY},
# {Y} year, {V} number, {K} top-k count.  The 5th template of each list is test-only.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "COUNT_ALL": {
        "en": ["how many {T} are there?", "count the number of {T}", "what is the total number of {T}?",
               "tell me the number of {T}", "how many {T} does the college have?"],
        "hi": ["कितने {T} हैं?", "{T_obl} की संख्या बताइए", "कुल कितने {T} हैं?",
               "{T_obl} की कुल संख्या क्या है?", "कॉलेज में कितने {T} हैं?"],
        "hg": ["kitne {T} hain?", "{T} ki sankhya batao", "total kitne {T} hain?",
               "{T} ka count kya hai?", "college mein kitne {T} hain?"],
    },
    "COUNT_DEPT": {
        "en": ["how many {T} are in the {D} department?", "count {T} of {D} department",
               "number of {T} in {D}", "how many {T} belong to {D} department?",
               "what is the count of {T} in the {D} department?"],
        "hi": ["{D} विभाग में कितने {T} हैं?", "{D} विभाग के {T_obl} की संख्या बताइए",
               "{D} में कितने {T} हैं?", "{D} विभाग में {T_obl} की गिनती करो",
               "{D} विभाग के कुल {T} कितने हैं?"],
        "hg": ["{D} department mein kitne {T} hain?", "{D} department ke {T} ki sankhya batao",
               "{D} mein kitne {T} hain?", "{D} department ke {T} count karo",
               "{D} vibhag mein total kitne {T} hain?"],
    },
    "COUNT_CITY": {
        "en": ["how many {T} are from {CITY}?", "count {T} who live in {CITY}",
               "number of {T} from {CITY}", "how many {T} belong to {CITY}?",
               "what is the count of {T} living in {CITY}?"],
        "hi": ["{CITY} से कितने {T} हैं?", "{CITY} के {T_obl} की संख्या बताइए",
               "कितने {T} {CITY} में रहते हैं?", "{CITY} के कितने {T} हैं?",
               "{CITY} में रहने वाले {T_obl} की गिनती क्या है?"],
        "hg": ["{CITY} se kitne {T} hain?", "{CITY} ke {T} ki sankhya batao",
               "kitne {T} {CITY} mein rehte hain?", "{CITY} wale {T} kitne hain?",
               "{CITY} mein rehne wale {T} ka count kya hai?"],
    },
    "LIST_ALL": {
        "en": ["show all {T}", "list all the {T}", "give me the names of all {T}",
               "list the names of {T}", "show me every one of the {T}"],
        "hi": ["सभी {T_obl} के नाम दिखाइए", "सारे {T} दिखाओ", "सभी {T_obl} की सूची बताइए",
               "{T_obl} के नाम बताओ", "सभी {T_obl} की सूची दिखाओ"],
        "hg": ["sabhi {T} dikhao", "saare {T} ke naam batao", "sab {T} ki list do",
               "{T} ke naam dikhao", "all {T} ki list dikhao"],
    },
    "LIST_CITY": {
        "en": ["list {T} from {CITY}", "show the {T} who live in {CITY}", "which {T} are from {CITY}?",
               "names of {T} from {CITY}", "who are the {T} living in {CITY}?"],
        "hi": ["{CITY} के {T_obl} के नाम दिखाइए", "{CITY} में रहने वाले {T} दिखाओ",
               "कौन से {T} {CITY} से हैं?", "{CITY} के {T_obl} की सूची बताइए",
               "{CITY} में कौन कौन से {T} रहते हैं?"],
        "hg": ["{CITY} ke {T} dikhao", "{CITY} mein rehne wale {T} ke naam batao",
               "kaun se {T} {CITY} se hain?", "{CITY} wale {T} ki list do",
               "{CITY} ke saare {T} kaun hain?"],
    },
    "LIST_DEPT": {
        "en": ["list {T} in the {D} department", "show {T} of {D} department", "which {T} are in {D}?",
               "names of {T} in the {D} department", "display all {T} belonging to {D} department"],
        "hi": ["{D} विभाग के {T} दिखाओ", "{D} विभाग के {T_obl} के नाम बताइए", "{D} में कौन से {T} हैं?",
               "{D} विभाग के {T_obl} की सूची दिखाइए", "{D} विभाग में कौन कौन से {T} हैं?"],
        "hg": ["{D} department ke {T} dikhao", "{D} department ke {T} ke naam batao",
               "{D} mein kaun se {T} hain?", "{D} department ke {T} ki list do",
               "{D} vibhag mein kaun kaun se {T} hain?"],
    },
    "LIST_YEAR": {
        "en": ["list {T} in {Y} year", "show {Y} year {T}", "which {T} are in {Y} year?",
               "names of {Y} year {T}", "who are the {T} studying in {Y} year?"],
        "hi": ["{Y} वर्ष के {T} दिखाओ", "{Y} साल के {T_obl} के नाम बताइए", "कौन से {T} {Y} वर्ष में हैं?",
               "{Y} वर्ष के {T_obl} की सूची दिखाइए", "{Y} साल में पढ़ने वाले {T} कौन हैं?"],
        "hg": ["{Y} year ke {T} dikhao", "{Y} saal ke {T} ke naam batao", "kaun se {T} {Y} year mein hain?",
               "{Y} year wale {T} ki list do", "{Y} saal mein padhne wale {T} kaun hain?"],
    },
    "LIST_GT": {
        "en": ["list {T} with {C} greater than {V}", "show {T} whose {C} is more than {V}",
               "which {T} have {C} above {V}?", "names of {T} having {C} over {V}",
               "find {T} where {C} exceeds {V}"],
        "hi": ["{V} से अधिक {C} वाले {T} दिखाओ", "जिन {T_obl} {ka} {C} {V} से ज़्यादा है उनके नाम बताइए",
               "किन {T_obl} {ka} {C} {V} से ऊपर है?", "{C} {V} से अधिक वाले {T_obl} की सूची दिखाइए",
               "{V} से ज़्यादा {C} वाले {T} कौन हैं?"],
        "hg": ["{V} se zyada {C} wale {T} dikhao", "jin {T} {ka} {C} {V} se zyada hai unke naam batao",
               "kin {T} {ka} {C} {V} se upar hai?", "{C} {V} se adhik wale {T} ki list do",
               "{V} se zyada {C} wale {T} kaun hain?"],
    },
    "LIST_LT": {
        "en": ["list {T} with {C} less than {V}", "show {T} whose {C} is lower than {V}",
               "which {T} have {C} below {V}?", "names of {T} having {C} under {V}",
               "find {T} where {C} is smaller than {V}"],
        "hi": ["{V} से कम {C} वाले {T} दिखाओ", "जिन {T_obl} {ka} {C} {V} से कम है उनके नाम बताइए",
               "किन {T_obl} {ka} {C} {V} से नीचे है?", "{C} {V} से कम वाले {T_obl} की सूची दिखाइए",
               "{V} से कम {C} वाले {T} कौन हैं?"],
        "hg": ["{V} se kam {C} wale {T} dikhao", "jin {T} {ka} {C} {V} se kam hai unke naam batao",
               "kin {T} {ka} {C} {V} se neeche hai?", "{C} {V} se kam wale {T} ki list do",
               "{V} se kam {C} wale {T} kaun hain?"],
    },
    "MAX_VAL": {
        "en": ["what is the highest {C} of {T}?", "maximum {C} among {T}", "what is the maximum {C} of all {T}?",
               "find the largest {C} of {T}", "tell me the top {C} value among {T}"],
        "hi": ["{T_obl} में सबसे अधिक {C} क्या है?", "{T_obl} {ka} अधिकतम {C} बताइए",
               "{T_obl} में अधिकतम {C} क्या है?", "सभी {T_obl} में सबसे ज़्यादा {C} बताओ",
               "{T_obl} में {C} का अधिकतम मान क्या है?"],
        "hg": ["{T} mein sabse zyada {C} kya hai?", "{T} {ka} maximum {C} batao",
               "{T} mein highest {C} kya hai?", "sabhi {T} mein sabse adhik {C} batao",
               "{T} ke {C} ki maximum value kya hai?"],
    },
    "MIN_VAL": {
        "en": ["what is the lowest {C} of {T}?", "minimum {C} among {T}", "what is the minimum {C} of all {T}?",
               "find the smallest {C} of {T}", "tell me the least {C} value among {T}"],
        "hi": ["{T_obl} में सबसे कम {C} क्या है?", "{T_obl} {ka} न्यूनतम {C} बताइए",
               "{T_obl} में न्यूनतम {C} क्या है?", "सभी {T_obl} में सबसे कम {C} बताओ",
               "{T_obl} में {C} का न्यूनतम मान क्या है?"],
        "hg": ["{T} mein sabse kam {C} kya hai?", "{T} {ka} minimum {C} batao",
               "{T} mein lowest {C} kya hai?", "sabhi {T} mein sabse kam {C} batao",
               "{T} ke {C} ki minimum value kya hai?"],
    },
    "AVG_VAL": {
        "en": ["what is the average {C} of {T}?", "find the mean {C} of {T}", "average {C} of all {T}",
               "tell me the average {C} among {T}", "what is the mean {C} for {T}?"],
        "hi": ["{T_obl} {ka} औसत {C} क्या है?", "{T_obl} {ka} औसत {C} बताइए", "सभी {T_obl} {ka} औसत {C} बताओ",
               "{T_obl} में औसत {C} क्या है?", "{T_obl} में {C} का औसत क्या है?"],
        "hg": ["{T} {ka} average {C} kya hai?", "{T} {ka} average {C} batao", "sabhi {T} {ka} average {C} batao",
               "{T} mein average {C} kya hai?", "{T} ke {C} ka average kitna hai?"],
    },
    "SUM_VAL": {
        "en": ["what is the total {C} of {T}?", "sum of {C} of all {T}", "calculate the sum of {C} for {T}",
               "total {C} of all {T}", "what is the combined {C} of {T}?"],
        "hi": ["{T_obl} {ka} कुल {C} क्या है?", "सभी {T_obl} {ka} कुल {C} बताइए", "{T_obl} में {C} का योग क्या है?",
               "सभी {T_obl} {ka} {C} जोड़कर बताओ", "{T_obl} {ka} संयुक्त {C} क्या है?"],
        "hg": ["{T} {ka} total {C} kya hai?", "sabhi {T} {ka} total {C} batao", "{T} ke {C} ka sum kya hai?",
               "{T} {ka} {C} jod ke batao", "saare {T} {ka} kul {C} kitna hai?"],
    },
    "ARGMAX": {
        "en": ["which {T_s} has the highest {C}?", "who has the highest {C} among {T}?",
               "name the {T_s} with the maximum {C}", "which {T_s} has the most {C}?",
               "find the {T_s} having the highest {C}"],
        "hi": ["किस {T_s} {ka} {C} सबसे अधिक है?", "{T_obl} में सबसे अधिक {C} किसका है?",
               "अधिकतम {C} वाले {T_s} का नाम बताइए", "सबसे ज़्यादा {C} किस {T_s} {ka} है?",
               "सबसे ज़्यादा {C} वाला {T_s} कौन है?"],
        "hg": ["kis {T_s} {ka} {C} sabse zyada hai?", "{T} mein sabse highest {C} kiska hai?",
               "maximum {C} wale {T_s} ka naam batao", "sabse adhik {C} kis {T_s} {ka} hai?",
               "sabse zyada {C} wala {T_s} kaun hai?"],
    },
    "ARGMIN": {
        "en": ["which {T_s} has the lowest {C}?", "who has the lowest {C} among {T}?",
               "name the {T_s} with the minimum {C}", "which {T_s} has the least {C}?",
               "find the {T_s} having the smallest {C}"],
        "hi": ["किस {T_s} {ka} {C} सबसे कम है?", "{T_obl} में सबसे कम {C} किसका है?",
               "न्यूनतम {C} वाले {T_s} का नाम बताइए", "सबसे कम {C} किस {T_s} {ka} है?",
               "सबसे कम {C} वाला {T_s} कौन है?"],
        "hg": ["kis {T_s} {ka} {C} sabse kam hai?", "{T} mein sabse lowest {C} kiska hai?",
               "minimum {C} wale {T_s} ka naam batao", "sabse kam {C} kis {T_s} {ka} hai?",
               "sabse kam {C} wala {T_s} kaun hai?"],
    },
    "TOP_K": {
        "en": ["top {K} {T} by {C}", "show the {K} {T} with the highest {C}", "list top {K} {T} based on {C}",
               "who are the top {K} {T} in terms of {C}?", "give me the {K} {T} having the highest {C}"],
        "hi": ["{C} के आधार पर शीर्ष {K} {T} दिखाओ", "सबसे ज़्यादा {C} वाले {K} {T} बताइए",
               "{C} में टॉप {K} {T} कौन हैं?", "सर्वाधिक {C} वाले {K} {T_obl} की सूची दिखाइए",
               "{C} के हिसाब से टॉप {K} {T} बताओ"],
        "hg": ["{C} ke basis par top {K} {T} dikhao", "sabse zyada {C} wale {K} {T} batao",
               "{C} mein top {K} {T} kaun hain?", "highest {C} wale {K} {T} ki list do",
               "{C} ke hisaab se top {K} {T} batao"],
    },
    "GROUP_COUNT": {
        "en": ["how many {T} are in each department?", "number of {T} per department",
               "count {T} in every department", "show the department wise count of {T}",
               "for each department, how many {T} are there?"],
        "hi": ["हर विभाग में कितने {T} हैं?", "प्रत्येक विभाग में {T_obl} की संख्या बताइए",
               "विभाग के अनुसार {T_obl} की गिनती दिखाओ", "हर विभाग के {T_obl} की संख्या दिखाइए",
               "प्रत्येक विभाग में कुल कितने {T} हैं?"],
        "hg": ["har department mein kitne {T} hain?", "har department ke {T} ki sankhya batao",
               "department wise {T} ka count dikhao", "pratyek vibhag mein kitne {T} hain?",
               "har ek department mein total kitne {T} hain?"],
    },
    "GROUP_AVG": {
        "en": ["average {C} of {T} in each department", "what is the average {C} of {T} per department?",
               "show department wise average {C} of {T}", "average {C} of {T} grouped by department",
               "for every department, find the mean {C} of {T}"],
        "hi": ["हर विभाग में {T_obl} {ka} औसत {C} क्या है?", "प्रत्येक विभाग के {T_obl} {ka} औसत {C} बताइए",
               "विभाग के अनुसार {T_obl} {ka} औसत {C} दिखाओ", "हर विभाग {ka} औसत {C} बताओ",
               "विभागवार {T_obl} {ka} औसत {C} दिखाइए"],
        "hg": ["har department mein {T} {ka} average {C} kya hai?", "har department ke {T} {ka} average {C} batao",
               "department wise average {C} dikhao", "har vibhag {ka} average {C} batao",
               "pratyek department ke {T} {ka} average {C} kya hai?"],
    },
}

KA = {"hi": {"m": "का", "f": "की"}, "hg": {"m": "ka", "f": "ki"}}
K_VALUES = [3, 5, 10]
K_WORDS = {"en": {3: "three", 5: "five", 10: "ten"},
           "hi": {3: "तीन", 5: "पाँच", 10: "दस"},
           "hg": {3: "teen", 5: "paanch", 10: "das"}}


def fmt_num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def slot_options(intent, lang):
    """Yield (surface_slots, sql_slots) for every combination of slot values."""
    spec = SLOTS[intent]
    if "c" in spec:
        tables_cols = [(c.split(".")[0], c) for c in spec["c"]]
    else:
        tables_cols = [(t, None) for t in spec["t"]]

    for table, col in tables_cols:
        t_forms = SCHEMA["tables"][table][lang]
        c_forms = SCHEMA["columns"][col][lang] if col else [None]
        extras = [{}]
        if spec.get("d"):
            extras = [{"d": d, "D": rng.choice(f[lang])} for d, f in SCHEMA["values"]["departments.name"].items()]
        elif spec.get("city"):
            extras = [{"city": c, "CITY": rng.choice(f[lang])} for c, f in SCHEMA["values"]["students.city"].items()]
        elif spec.get("y"):
            extras = [{"y": y, "Y": rng.choice(f[lang])} for y, f in SCHEMA["values"]["students.year"].items()]
        elif spec.get("v"):
            extras = [{"v": fmt_num(v), "V": fmt_num(v)} for v in SCHEMA["thresholds"][col]]
        elif spec.get("k"):
            extras = []
            for k in K_VALUES:
                word = K_WORDS[lang][k] if rng.random() < 0.3 else str(k)
                extras.append({"k": k, "K": word})

        for tf, cf, ex in itertools.product(t_forms, c_forms, extras):
            surface = {"T": tf["dir"], "T_obl": tf.get("obl", tf["dir"]), "T_s": tf["sing"], **ex}
            if cf:
                surface["C"] = cf["w"]
                if lang in KA:
                    surface["ka"] = KA[lang][cf["g"]]
            sql_slots = {"t": table, "c": col.split(".")[1] if col else None, **ex}
            yield surface, sql_slots


def add_noise(question):
    """Swap some Hinglish words for common misspellings (Hinglish has no fixed spelling)."""
    words = question.split()
    out = []
    for w in words:
        core = w.rstrip("?")
        tail = w[len(core):]
        if core in VARIANTS and rng.random() < NOISE_PROB:
            core = rng.choice(VARIANTS[core])
        out.append(core + tail)
    return " ".join(out)


def build():
    items = []
    for intent, by_lang in TEMPLATES.items():
        for lang, templates in by_lang.items():
            options = list(slot_options(intent, lang))
            for ti, template in enumerate(templates):
                split = "test" if ti == len(templates) - 1 else "train"
                for surface, sql_slots in rng.sample(options, k=min(FILLS_PER_TEMPLATE, len(options))):
                    q = template.format(**surface)
                    if lang == "hg":
                        q = add_noise(q)
                    sql = SQL[intent].format(**sql_slots)
                    items.append({"lang": lang, "intent": intent, "question": q, "sql": sql,
                                  "template": f"{intent}/{lang}/{ti}", "split": split})

    # De-duplicate identical questions, then assign ids
    seen, unique = set(), []
    for it in items:
        key = (it["lang"], it["question"])
        if key not in seen:
            seen.add(key)
            unique.append(it)
    for i, it in enumerate(unique, 1):
        it["id"] = i

    OUT.mkdir(exist_ok=True)
    train = [it for it in unique if it["split"] == "train"]
    test = [it for it in unique if it["split"] == "test"]
    for name, data in [("train", train), ("test", test)]:
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return train, test


def build_wild():
    """60 hand-written questions (data/dataset/wild_source.tsv): free phrasing, never used for training/tuning."""
    import sqlite3
    con = sqlite3.connect(HERE / "college.db")
    items = []
    for line in (OUT / "wild_source.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        lang, q, sql = line.split("	")
        con.execute(sql).fetchall()  # every gold query must run
        items.append({"id": 10000 + len(items), "lang": lang, "intent": None, "question": q, "sql": sql,
                      "template": "wild", "split": "wild"})
    con.close()
    (OUT / "wild.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


rng = random.Random(7)

if __name__ == "__main__":
    train, test = build()
    print(f"wild={len(build_wild())}")
    from collections import Counter
    print(f"train={len(train)} test={len(test)}")
    print("train by lang:", dict(Counter(i["lang"] for i in train)))
    print("test by lang: ", dict(Counter(i["lang"] for i in test)))
