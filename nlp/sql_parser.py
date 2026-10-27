"""Phase 4 — PARSer-lite: parse a SQL query into its clauses.

PARSQL's PARSer walks the SQL syntax tree to (1) split a query into sub-SQL fragments and
(2) write a step-by-step explanation of it. This is a small version for the SQL subset our
system produces: SELECT [agg] ... FROM t [JOIN ... ON ...] [WHERE a AND b] [GROUP BY]
[ORDER BY c ASC|DESC] [LIMIT k].

The parsed form is also used in evaluation to compare a predicted query with the gold one
clause by clause (which table, which aggregation, which conditions...).
"""
import re

CLAUSE_RE = re.compile(
    r"^SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>\w+)"
    r"(?:\s+JOIN\s+(?P<join>\w+)\s+ON\s+(?P<on>[\w.]+\s*=\s*[\w.]+))?"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+GROUP\s+BY\s+(?P<group>[\w.]+))?"
    r"(?:\s+ORDER\s+BY\s+(?P<order>[\w.]+)(?:\s+(?P<dir>ASC|DESC))?)?"
    r"(?:\s+LIMIT\s+(?P<limit>\d+))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
AGG_RE = re.compile(r"^(COUNT|MAX|MIN|AVG|SUM)\((.+)\)$", re.IGNORECASE)
COND_RE = re.compile(r"^([\w.]+)\s*(=|>=|<=|>|<|!=)\s*(.+)$")


def _bare(col):
    """students.cgpa -> cgpa"""
    return col.split(".")[-1].strip()


def _split_and(where):
    parts, depth, cur, i = [], 0, "", 0
    tokens = re.split(r"(\s+AND\s+)", where, flags=re.IGNORECASE)
    for t in tokens:
        if re.fullmatch(r"\s+AND\s+", t, flags=re.IGNORECASE) and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += t
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse(sql):
    """Return a dict of clauses, or raise ValueError for SQL outside the supported subset."""
    m = CLAUSE_RE.match(sql.strip())
    if not m:
        raise ValueError(f"unsupported SQL: {sql}")
    select = []
    for item in [s.strip() for s in m.group("select").split(",")]:
        a = AGG_RE.match(item)
        if a:
            select.append({"agg": a.group(1).upper(), "column": _bare(a.group(2))})
        else:
            select.append({"agg": None, "column": item if "." not in item else item})
    conditions = []
    if m.group("where"):
        for c in _split_and(m.group("where")):
            cm = COND_RE.match(c)
            if not cm:
                raise ValueError(f"unsupported condition: {c}")
            col, op, val = cm.groups()
            val = val.strip()
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            else:
                val = float(val) if "." in val else int(val)
            conditions.append({"column": col, "op": op, "value": val})
    return {
        "select": select,
        "from": m.group("from"),
        "join": m.group("join"),
        "on": m.group("on"),
        "where": conditions,
        "group_by": m.group("group"),
        "order_by": m.group("order"),
        "direction": (m.group("dir") or "ASC").upper() if m.group("order") else None,
        "limit": int(m.group("limit")) if m.group("limit") else None,
    }


def fragments(sql):
    """Split a query into sub-SQL fragments (PARSer-style), one per clause."""
    p = parse(sql)
    out = [f"FROM {p['from']}"]
    if p["join"]:
        out.append(f"JOIN {p['join']} ON {p['on']}")
    for c in p["where"]:
        v = f"'{c['value']}'" if isinstance(c["value"], str) else c["value"]
        out.append(f"WHERE {c['column']} {c['op']} {v}")
    if p["group_by"]:
        out.append(f"GROUP BY {p['group_by']}")
    sel = ", ".join(f"{s['agg']}({s['column']})" if s["agg"] else s["column"] for s in p["select"])
    out.append(f"SELECT {sel}")
    if p["order_by"]:
        out.append(f"ORDER BY {p['order_by']} {p['direction']}")
    if p["limit"] is not None:
        out.append(f"LIMIT {p['limit']}")
    return out


def components(sql):
    """Normalized clause signature used for component-match accuracy and error analysis."""
    p = parse(sql)
    return {
        "table": p["from"],
        "aggregation": tuple(sorted(s["agg"] or "-" for s in p["select"])),
        "columns": tuple(sorted(_bare(s["column"]) for s in p["select"])),
        "conditions": tuple(sorted((_bare(c["column"]), c["op"], str(c["value"]).lower()) for c in p["where"])),
        "group_by": _bare(p["group_by"]) if p["group_by"] else None,
        "order": (_bare(p["order_by"]), p["direction"]) if p["order_by"] else None,
        "limit": p["limit"],
    }


def normalize(sql):
    """Canonical string for exact-match comparison (whitespace, case of keywords, trailing ;)."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()
