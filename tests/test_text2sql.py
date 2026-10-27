import pytest

from nlp.corpora import dataset
from nlp.reasoning import explain
from nlp.sql_parser import components, fragments, parse
from nlp.text2sql import build_sql, execute, run

CASES = [
    ("computer dept me kitney studnts h?",
     "SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id WHERE departments.name = 'Computer'"),
    ("How many students are in the Computer department?",
     "SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id WHERE departments.name = 'Computer'"),
    ("कंप्यूटर विभाग में कितने छात्र हैं?",
     "SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id WHERE departments.name = 'Computer'"),
    ("top 3 teachers by salary", "SELECT name FROM faculty ORDER BY salary DESC LIMIT 3"),
    ("average cgpa of students from pune", "SELECT AVG(cgpa) FROM students WHERE city = 'Pune'"),
    ("har department mein kitne students hain",
     "SELECT departments.name, COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id GROUP BY departments.name"),
    ("9 से ज़्यादा सीजीपीए वाले छात्रों के नाम", "SELECT name FROM students WHERE cgpa > 9"),
]


@pytest.mark.parametrize("question,gold", CASES)
def test_pipeline_examples(question, gold):
    assert run(question, execute_sql=False)["sql"] == gold


def test_compound_question_gets_two_filters():
    sql = run("students with cgpa above 8.5 in mechanical department", execute_sql=False)["sql"]
    p = parse(sql)
    assert {(c["column"].split(".")[-1], c["op"]) for c in p["where"]} == {("name", "="), ("cgpa", ">")}


def test_off_topic_question_is_refused():
    t = run("hello how are you")
    assert t["sql"] is None and "doesn't mention" in t["error"]


def test_every_gold_query_parses_and_explains():
    for it in dataset("train") + dataset("test") + dataset("wild"):
        parse(it["sql"])
        assert fragments(it["sql"])
        assert explain(it["sql"], it["lang"])


def test_builder_reproduces_gold_format():
    assert build_sql("LIST", "faculty", filters=[{"kind": "dept", "value": "IT"}]) == \
        "SELECT faculty.name FROM faculty JOIN departments ON faculty.dept_id = departments.id WHERE departments.name = 'IT'"
    assert build_sql("ARGMIN", "students", "fees") == "SELECT name FROM students ORDER BY fees ASC LIMIT 1"
    assert components("SELECT MAX(cgpa) FROM students")["aggregation"] == ("MAX",)


def test_database_is_read_only():
    with pytest.raises(Exception):
        execute("DELETE FROM students")
    assert execute("SELECT COUNT(*) FROM students")["rows"] == [[120]]
