"""Guardrail retry-loop tests, with a mock backend (no model, no network).

Run:  python -m scripts.test_guardrail
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.guardrail import generate_grounded


class MockBackend:
    """Returns scripted answers in order; repeats the last one if asked again."""
    name = "mock"
    model = "mock"

    def __init__(self, answers):
        self._answers = answers
        self.calls = 0

    def generate(self, system, user, max_tokens=2048):
        ans = self._answers[min(self.calls, len(self._answers) - 1)]
        self.calls += 1
        return ans


def test_passes_first_try_when_grounded():
    b = MockBackend(["Grounded claim [1106.4789]."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"])
    assert rep["grounded"] and rep["problem"] is None and rep["attempts"] == 1 and b.calls == 1


def test_fabrication_retries_then_succeeds():
    b = MockBackend(["Bad [hep-th/9999999].", "Fixed [1106.4789]."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"], max_retries=2)
    assert rep["grounded"] and rep["attempts"] == 2 and b.calls == 2 and "1106.4789" in ans


def test_fabrication_gives_up_after_max_retries():
    b = MockBackend(["Always bad [hep-th/9999999]."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"], max_retries=2)
    assert not rep["grounded"] and rep["problem"] == "fabricated" and rep["attempts"] == 3 and b.calls == 3


def test_refusal_with_no_citations_passes():
    b = MockBackend(["The passages do not support an answer to this question."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"])
    assert rep["grounded"] and rep["validation"]["uncited"] and b.calls == 1


def test_uncited_substantive_triggers_retry():
    # The small-model failure mode: a confident prose answer with no citations.
    b = MockBackend(["Chern-Simons theory yields the Jones polynomial.", "Now grounded [1106.4789]."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"], max_retries=2)
    assert rep["grounded"] and rep["attempts"] == 2 and b.calls == 2


def test_uncited_substantive_gives_up():
    b = MockBackend(["Chern-Simons theory yields the Jones polynomial."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"], max_retries=2)
    assert not rep["grounded"] and rep["problem"] == "uncited" and b.calls == 3


def test_require_citation_false_allows_uncited():
    # Out-of-corpus probe: a correct refusal/short answer should not be forced to cite.
    b = MockBackend(["This is outside the scope of the provided material."])
    ans, rep = generate_grounded(b, "sys", "user", ["1106.4789"], require_citation=False)
    assert rep["grounded"] and b.calls == 1


def test_prerefusal_text_counts_as_refusal():
    # The rerank pre-generation refusal text must match the guardrail's refusal
    # pattern, so it is never mistaken for an uncited substantive answer.
    from scripts.guardrail import _REFUSAL_RE
    from scripts.rerank import REFUSAL_TEXT, refusal_check

    assert _REFUSAL_RE.search(REFUSAL_TEXT.format(score=-1.23))
    assert refusal_check([(-0.5, {})]) == (True, -0.5)
    assert refusal_check([(0.5, {})]) == (False, 0.5)
    assert refusal_check([]) == (True, float("-inf"))


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _run()
