"""Phase 4 — Text-to-reason: a step-by-step explanation of a SQL query, in English, Hindi or Hinglish.

This mirrors the rule-based half of PARSQL's reasoning generation: every clause of the parsed
query (nlp.sql_parser) becomes one explanation step written with the schema's words in the
question's language. Each step also lists the common ways people phrase that operation
("how many / number of / count"), so the explanation can be compared with the question in the
selection step (nlp.text2sql).
"""
from nlp.schema import schema
from nlp.sql_parser import parse

T = {
    "en": {
        "from": "Look in the {T} table.",
        "join": "Join each {Ts} with its department.",
        "dept": "Keep only {T} of the {V} department.",
        "city": "Keep only {T} from {V}, living in {V}.",
        "year": "Keep only {T} in {V} year.",
        ">": "Keep {T} whose {C} is greater than, more than, above, over {V}.",
        "<": "Keep {T} whose {C} is less than, lower than, below, under {V}.",
        "group": "Group by department: each, every, per department, department wise.",
        "COUNT": "Count them: how many, number of, count, total number of {T}.",
        "LIST": "Show, list, display, find the names of the {T}: which, who are they.",
        "MAX": "Find the highest, maximum, largest, top {C} value.",
        "MIN": "Find the lowest, minimum, smallest, least {C} value.",
        "AVG": "Find the average, mean {C}.",
        "SUM": "Find the total, sum, combined {C}, adding all.",
        "DESC1": "Sort by {C} from highest to lowest and take the single {Ts} with the highest, maximum, most {C}: which, who has it.",
        "ASC1": "Sort by {C} from lowest to highest and take the single {Ts} with the lowest, minimum, least, smallest {C}: which, who has it.",
        "TOPK": "Sort by {C}, highest first, and take the top {K} {T}.",
        "GROUP_COUNT": "Count how many, number of {T} in each department.",
        "GROUP_AVG": "Find the average, mean {C} of {T} in each department.",
    },
    "hg": {
        "from": "{T} table mein dekho.",
        "join": "har {Ts} ka department jodo.",
        "dept": "sirf {V} department ke {T} rakho.",
        "city": "sirf {V} ke, {V} se, {V} mein rehne wale {T} rakho.",
        "year": "sirf {V} saal, {V} year ke {T} rakho.",
        ">": "jinka {C} {V} se zyada, adhik, upar hai woh {T} rakho.",
        "<": "jinka {C} {V} se kam, neeche hai woh {T} rakho.",
        "group": "har department ke hisaab se group karo: har, pratyek department, department wise.",
        "COUNT": "kitne {T} hain gino: count, sankhya, total kitne.",
        "LIST": "{T} ke naam dikhao, batao, list do: kaun kaun hain.",
        "MAX": "sabse zyada, maximum, highest, adhik {C} kya hai nikalo.",
        "MIN": "sabse kam, minimum, lowest {C} kya hai nikalo.",
        "AVG": "average, avg, ausat {C} kya hai nikalo.",
        "SUM": "total, sum, kul {C} jod ke batao.",
        "DESC1": "{C} ke hisaab se sort karo, sabse zyada, highest, maximum {C} wala {Ts} kaun hai, kiska, kis.",
        "ASC1": "{C} ke hisaab se sort karo, sabse kam, lowest, minimum {C} wala {Ts} kaun hai, kiska, kis.",
        "TOPK": "{C} ke basis par, hisaab se top {K} {T} dikhao.",
        "GROUP_COUNT": "har department mein kitne {T} hain, count, sankhya.",
        "GROUP_AVG": "har department mein {T} ka average {C}.",
    },
    "hi": {
        "from": "{T} तालिका देखें।",
        "join": "हर {Ts} का विभाग जोड़ें।",
        "dept": "केवल {V} विभाग के {Tobl} को रखें।",
        "city": "केवल {V} के, {V} से, {V} में रहने वाले {Tobl} को रखें।",
        "year": "केवल {V} वर्ष, {V} साल के {T} रखें।",
        ">": "जिनका {C} {V} से अधिक, ज़्यादा, ऊपर है वे {T} रखें।",
        "<": "जिनका {C} {V} से कम, नीचे है वे {T} रखें।",
        "group": "हर विभाग, प्रत्येक विभाग के अनुसार समूह बनाएं, विभागवार।",
        "COUNT": "कितने {T} हैं गिनें: संख्या, गिनती, कुल कितने।",
        "LIST": "{Tobl} के नाम, सूची दिखाएं, बताएं: कौन कौन हैं।",
        "MAX": "सबसे अधिक, सबसे ज़्यादा, अधिकतम {C} क्या है निकालें।",
        "MIN": "सबसे कम, न्यूनतम {C} क्या है निकालें।",
        "AVG": "औसत {C} क्या है निकालें।",
        "SUM": "कुल {C}, योग, जोड़कर बताएं।",
        "DESC1": "{C} के अनुसार क्रम दें, सबसे अधिक, सबसे ज़्यादा, अधिकतम {C} वाला {Ts} कौन है, किसका, किस।",
        "ASC1": "{C} के अनुसार क्रम दें, सबसे कम, न्यूनतम {C} वाला {Ts} कौन है, किसका, किस।",
        "TOPK": "{C} के आधार पर, हिसाब से शीर्ष, टॉप {K} {T} दिखाएं।",
        "GROUP_COUNT": "हर विभाग में कितने {T} हैं, संख्या, गिनती।",
        "GROUP_AVG": "हर विभाग में {Tobl} का औसत {C}।",
    },
}


def _forms(lang):
    sc = schema()
    tables = {t: v[lang][0] for t, v in sc["tables"].items()}
    cols = {c.split(".")[1]: v[lang][0]["w"] for c, v in sc["columns"].items()}
    vals = {}
    for col, values in sc["values"].items():
        for value, langs in values.items():
            vals[(col.split(".")[1] if col != "departments.name" else "dept", value)] = langs[lang][0]
    return tables, cols, vals


def _bare(c):
    return c.split(".")[-1]


def explain(sql, lang="en"):
    """Return a list of explanation steps for a SQL query."""
    lang = lang if lang in T else "en"
    tpl = T[lang]
    tables, cols, vals = _forms(lang)
    p = parse(sql)
    t = tables.get(p["from"], {"dir": p["from"], "sing": p["from"]})
    ctx = {"T": t["dir"], "Ts": t["sing"], "Tobl": t.get("obl", t["dir"])}

    def fill(key, **extra):
        return tpl[key].format(**{**ctx, **extra})

    steps = [fill("from")]
    if p["join"] and not p["group_by"] and not any(_bare(c["column"]) == "name" and "departments" in c["column"]
                                                    for c in p["where"]):
        steps.append(fill("join"))
    if p["join"] and p["group_by"]:
        steps.append(fill("join"))
    for c in p["where"]:
        col = _bare(c["column"])
        if "departments" in c["column"]:
            steps.append(fill("dept", V=vals.get(("dept", c["value"]), c["value"])))
        elif col == "city":
            steps.append(fill("city", V=vals.get(("city", c["value"]), c["value"])))
        elif col == "year":
            steps.append(fill("year", V=vals.get(("year", str(c["value"])), c["value"])))
        elif c["op"] in (">", ">="):
            steps.append(fill(">", C=cols.get(col, col), V=c["value"]))
        elif c["op"] in ("<", "<="):
            steps.append(fill("<", C=cols.get(col, col), V=c["value"]))
    if p["group_by"]:
        steps.append(fill("group"))
    aggs = [s for s in p["select"] if s["agg"]]
    if p["group_by"]:
        agg = aggs[0] if aggs else None
        if agg and agg["agg"] == "AVG":
            steps.append(fill("GROUP_AVG", C=cols.get(_bare(agg["column"]), agg["column"])))
        else:
            steps.append(fill("GROUP_COUNT"))
    elif aggs:
        a = aggs[0]
        if a["agg"] == "COUNT":
            steps.append(fill("COUNT"))
        else:
            steps.append(fill(a["agg"], C=cols.get(_bare(a["column"]), a["column"])))
    elif p["order_by"]:
        C = cols.get(_bare(p["order_by"]), p["order_by"])
        if p["limit"] == 1:
            steps.append(fill("DESC1" if p["direction"] == "DESC" else "ASC1", C=C))
        else:
            steps.append(fill("TOPK", C=C, K=p["limit"]))
    else:
        steps.append(fill("LIST"))
    return steps
