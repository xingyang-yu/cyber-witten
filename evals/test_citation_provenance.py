"""Fast, dependency-free, network-free checks for citation provenance.

The classifier's local tiers (date-plausibility, in-corpus, digit-slip) are
tested directly; the INSPIRE tier is exercised through a pre-seeded cache, so
this runs with no index file and no network.

Run:  python -m evals.test_citation_provenance      (plain asserts, no pytest)
"""
from evals.citation_provenance import (
    BAD_DATE,
    IN_CORPUS,
    NO_RECORD,
    OUT_REAL,
    UNKNOWN,
    classify,
    date_plausible,
    digit_slip_of,
)


def test_date_plausible_valid_windows():
    for pid in ["hep-th/9407087", "hep-th/9812208", "math-ph/0212366",
                "1803.04574", "2507.06945", "2206.10780"]:
        assert date_plausible(pid) is True, pid


def test_date_plausible_impossible_dates():
    assert date_plausible("hep-th/8910145") is False   # 1989: predates arXiv
    assert date_plausible("hep-th/9413001") is False    # month 13
    assert date_plausible("9912.10000") is False        # 2099: future-dated
    assert date_plausible("hep-th/0805001") is False     # old scheme ended 2007-03


def test_date_plausible_none_when_no_date():
    assert date_plausible("inspire:193975") is None


def test_digit_slip_detects_corpus_neighbor():
    corpus = {"2206.10780", "hep-th/9407087"}
    assert digit_slip_of("2306.10780", corpus) == "2206.10780"   # year digit slip
    assert digit_slip_of("2206.10790", corpus) == "2206.10780"   # last-digit slip
    assert digit_slip_of("1234.56789", corpus) is None           # no neighbor


def test_classify_local_tiers_need_no_network():
    corpus = {"hep-th/9407087"}
    # in-corpus real paper, resolved before any INSPIRE call
    assert classify("hep-th/9407087", corpus, {}, None, offline=False)[0] == IN_CORPUS
    # impossible date, resolved before any INSPIRE call
    assert classify("hep-th/8910145", corpus, {}, None, offline=False)[0] == BAD_DATE
    # out-of-corpus, plausible date, no network -> unknown (not a false verdict)
    assert classify("1601.03987", corpus, {}, None, offline=True)[0] == UNKNOWN


def test_classify_digit_slip_note_on_fabrication():
    corpus = {"2206.10780"}
    cache = {"2306.10780": {"exists": False, "title": None, "source": "inspire"}}
    verdict, note = classify("2306.10780", corpus, cache, None, offline=False)
    assert verdict == NO_RECORD
    assert note == "digit-slip of 2206.10780 (in corpus)"


def test_reproduces_hand_audit_8_real_7_fabricated():
    """The 15 flagged IDs from citation_audit.md, via a seeded INSPIRE cache."""
    corpus = {"hep-th/9407087"}                      # the one in-corpus real paper
    real_out = ["hep-th/9109055", "hep-th/9804195", "hep-th/9807022",
                "hep-th/9606101", "hep-th/9812208", "1803.04574", "2507.06945"]
    fake_rec = ["1605.08291", "1601.03987", "hep-th/9205140",
                "1803.04576", "math-ph/0212366", "2206.10790"]
    cache = {}
    for pid in real_out:
        cache[pid] = {"exists": True, "title": "x", "source": "inspire"}
    for pid in fake_rec:
        cache[pid] = {"exists": False, "title": None, "source": "inspire"}
    # hep-th/8910145 is caught locally by the date tier (no cache entry needed).
    all_ids = ["hep-th/9407087"] + real_out + fake_rec + ["hep-th/8910145"]

    real = fab = 0
    for pid in all_ids:
        verdict = classify(pid, corpus, cache, None, offline=False)[0]
        if verdict in (IN_CORPUS, OUT_REAL):
            real += 1
        elif verdict in (BAD_DATE, NO_RECORD):
            fab += 1
    assert (real, fab) == (8, 7), (real, fab)


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _run()
