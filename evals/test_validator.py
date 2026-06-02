"""Fast, dependency-free checks for the citation validator.

Run:  python -m evals.test_validator      (plain asserts, no pytest needed)
"""
from evals.validator import extract_citations, normalize_id, validate_citations


def test_extracts_three_id_forms():
    a = "Result A [1106.4789] follows from [hep-th/9112056] and [inspire:193975]."
    assert extract_citations(a) == ["1106.4789", "hep-th/9112056", "inspire:193975"]


def test_ignores_noise_brackets():
    # [cite] is passage-text noise; [outside corpus] is the speculation marker;
    # [2.7] is an equation ref. None are citations.
    a = "As shown [cite], see eq [2.7]; speculatively [outside corpus] this holds."
    assert extract_citations(a) == []


def test_splits_grouped_citations():
    a = "Both papers agree [1106.4789, 1101.3216; hep-th/9112056]."
    assert extract_citations(a) == ["1106.4789", "1101.3216", "hep-th/9112056"]


def test_version_suffix_normalized():
    assert normalize_id("HEP-TH/9112056V2") == "hep-th/9112056"
    a = "See [1106.4789v3]."
    res = validate_citations(a, retrieved_ids=["1106.4789"])
    assert res["invalid_citations"] == []
    assert not res["grounding_violation"]


def test_grounding_violation_on_unretrieved_cite():
    a = "Claim [1234.56789] not in context; claim [1106.4789] is."
    res = validate_citations(a, retrieved_ids=["1106.4789", "hep-th/9112056"])
    assert res["invalid_citations"] == ["1234.56789"]
    assert res["valid_citations"] == ["1106.4789"]
    assert res["grounding_violation"] is True


def test_uncited_answer_flagged():
    res = validate_citations("The passages do not address this.", retrieved_ids=["1106.4789"])
    assert res["uncited"] is True
    assert res["grounding_violation"] is False  # no citation can't be a *bad* citation


def test_outside_corpus_marker_detected():
    res = validate_citations("Grounded [1106.4789]; beyond that [outside corpus].",
                             retrieved_ids=["1106.4789"])
    assert res["marked_outside_corpus"] is True
    assert res["grounding_violation"] is False


def test_expected_recall_separates_retrieval_from_generation():
    # Right paper retrieved but model cited a different (also-retrieved) one:
    # retrieval_recall=1 (found it) but expected_citation_recall=0 (didn't cite it).
    res = validate_citations(
        "Answer [1101.3216].",
        retrieved_ids=["1106.4789", "1101.3216"],
        expected_ids=["1106.4789"],
    )
    assert res["retrieval_recall"] == 1.0
    assert res["expected_citation_recall"] == 0.0
    assert res["grounding_violation"] is False


def test_out_of_corpus_recall_is_none():
    res = validate_citations("No support in passages.", retrieved_ids=["1106.4789"],
                             expected_ids=[])
    assert res["expected_citation_recall"] is None
    assert res["retrieval_recall"] is None


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _run()
