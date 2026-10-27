import pytest

from nlp.transformer_intent import available

pytestmark = pytest.mark.skipif(not available(), reason="PyTorch not installed")


def test_transformer_ranks_all_intents():
    from nlp.transformer_intent import get_classifier

    ranked = get_classifier().rank(["how", "many", "students", "are", "there", "?"])
    assert ranked[0]["intent"] == "COUNT_ALL"
    assert abs(sum(r["prob"] for r in ranked) - 1) < 1e-4


def test_pipeline_with_transformer_intent():
    from nlp.text2sql import Options, run

    sql = run(
        "computer dept me kitney studnts h?",
        options=Options(intent_model="transformer"),
        execute_sql=False,
    )["sql"]
    assert sql == (
        "SELECT COUNT(*) FROM students JOIN departments ON students.dept_id = departments.id "
        "WHERE departments.name = 'Computer'"
    )
