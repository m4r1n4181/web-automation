from src.fuzzy_matcher import normalize_name, get_candidates


def test_normalize_name_basic():
    assert normalize_name("Zora Radić") == "ZORA RADIC"
    assert normalize_name("") == ""


def test_normalize_name_prefix_removal():
    assert normalize_name("UD TEST COMPANY") == "TEST COMPANY"
    assert normalize_name("DOO ALFA") == "ALFA"


def test_fuzzy_returns_results():
    results = get_candidates(
        pdf_name="PERIC PERA",
        filijala="Istocno Sarajevo",
        top_n=3
    )

    assert isinstance(results, list)
    assert len(results) <= 3
    assert "f_wratio" in results[0]


def test_fuzzy_ordering():
    results = get_candidates(
        pdf_name="PERIC PERA",
        filijala="Istocno Sarajevo",
        top_n=3
    )

    # rang = WRatio + bonus filijale (tie-break ne mijenja glavni ključ)
    scores = [r["f_wratio"] + r["branch_bonus"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_branch_bonus_only_for_matching_branch():
    # bez filijale nema bonusa
    results = get_candidates(pdf_name="PERIC PERA", filijala=None, top_n=5)
    assert all(r["branch_bonus"] == 0 for r in results)

    # bonus dobijaju samo kandidati iz zadate filijale
    results = get_candidates(pdf_name="PERIC PERA", filijala="Prijedor", top_n=5)
    for r in results:
        if r["branch_bonus"]:
            assert r["filijala"].upper() == "PRIJEDOR"