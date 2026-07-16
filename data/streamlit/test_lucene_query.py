"""find_seed_entities의 Lucene 쿼리 변환 셀프체크. 실행: pytest 또는 python."""
from main import _lucene_query


def test_lucene_query():
    assert _lucene_query("AirAsia Zest") == "AirAsia* Zest*"
    assert _lucene_query("Roketsan") == "Roketsan*"
    assert _lucene_query("A:B (x)") == r"A\:B* \(x\)*"  # Lucene 특수문자 이스케이프
    assert _lucene_query("   ") == ""
    assert _lucene_query("") == ""


if __name__ == "__main__":
    test_lucene_query()
    print("PASS")
