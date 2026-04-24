import logging
import sys
import pytest
from grid07.router   import PersonaRouter, RouteMatch
from grid07.personas import BOTS


router = PersonaRouter()


POSTS = {
    "ai_tech"   : "OpenAI released a new model replacing junior developers. Elon Musk says AGI is near.",
    "finance"   : "Fed raised interest rates 50 basis points. S&P 500 ETF futures down. Hedge funds repositioning.",
    "monopoly"  : "Amazon and Google buying every startup killing competition. Surveillance capitalism out of control.",
    "crypto_fin": "Bitcoin ETF hits all time high as Wall Street hedge funds pour money into crypto.",
    "unrelated" : "The weather today is sunny and warm. I went for a walk in the park.",
}

T = 0.20   # working threshold for TF-IDF


def ids(post: str, threshold: float = T) -> list[str]:
    return [m.bot.id for m in router.get_matches(post, threshold=threshold)]



@pytest.mark.parametrize("post_key, expected_bot", [
    ("ai_tech",    "Bot_A"),
    ("finance",    "Bot_C"),
    ("monopoly",   "Bot_B"),
    ("crypto_fin", "Bot_C"),
])
def test_correct_bot_matched(post_key, expected_bot):
    assert expected_bot in ids(POSTS[post_key])


def test_unrelated_matches_nobody():
    assert ids(POSTS["unrelated"]) == []


def test_finance_does_not_bleed_to_wrong_bots():
    matched = ids(POSTS["finance"])
    assert "Bot_A" not in matched
    assert "Bot_B" not in matched


#  2  RETURN CONTRACT


def test_route_returns_entry_for_every_bot():
    results = router.route(POSTS["ai_tech"], threshold=0.0)
    assert len(results) == len(BOTS)


def test_results_sorted_descending():
    results = router.route(POSTS["ai_tech"], threshold=0.0)
    scores  = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_routematch_types_and_range():
    for r in router.route(POSTS["finance"], threshold=0.0):
        assert isinstance(r, RouteMatch)
        assert isinstance(r.bot.id,  str)
        assert isinstance(r.score,   float)
        assert isinstance(r.matched, bool)
        assert 0.0 <= r.score <= 1.0


def test_get_matches_is_filtered_subset_of_route():
    results = router.route(POSTS["ai_tech"], threshold=T)
    matched = router.get_matches(POSTS["ai_tech"], threshold=T)
    assert matched == [r for r in results if r.matched]


# § 3  THRESHOLD BEHAVIOUR

def test_threshold_zero_matches_all_nonzero_bots():
    results = router.get_matches(POSTS["ai_tech"], threshold=0.0)
    assert any(r.bot.id == "Bot_A" for r in results)


def test_threshold_one_matches_nobody():
    assert router.get_matches(POSTS["ai_tech"], threshold=1.0) == []


def test_lower_threshold_yields_more_or_equal_matches():
    low  = router.get_matches(POSTS["ai_tech"], threshold=0.05)
    high = router.get_matches(POSTS["ai_tech"], threshold=0.90)
    assert len(low) >= len(high)



#  4  EDGE CASES

@pytest.mark.parametrize("post", [
    "",                          # empty
    "bitcoin " * 500,            # very long
    " OpenAI!!! #AGI $$$ ",  # special chars / emoji
    "AI",                        # single short word
])
def test_edge_case_does_not_crash(post):
    results = router.route(post, threshold=0.0)
    assert len(results) == len(BOTS)
    assert all(isinstance(r.score, float) for r in results)




if __name__ == "__main__":
    import sys

    suites = {
        # 1 routing accuracy
        "AI/Tech → Bot_A"              : lambda: test_correct_bot_matched("ai_tech",    "Bot_A"),
        "Finance → Bot_C"              : lambda: test_correct_bot_matched("finance",    "Bot_C"),
        "Monopoly → Bot_B"             : lambda: test_correct_bot_matched("monopoly",   "Bot_B"),
        "Crypto+Finance → Bot_C"       : lambda: test_correct_bot_matched("crypto_fin", "Bot_C"),
        "Unrelated → nobody"           : test_unrelated_matches_nobody,
        "Finance ≠ Bot_A / Bot_B"      : test_finance_does_not_bleed_to_wrong_bots,
        #  2 return contract
        "route() covers all bots"      : test_route_returns_entry_for_every_bot,
        "results sorted descending"    : test_results_sorted_descending,
        "RouteMatch types + range"     : test_routematch_types_and_range,
        "get_matches ⊆ route()"        : test_get_matches_is_filtered_subset_of_route,
        #  3 threshold
        "threshold=0.0 → nonzero hit"  : test_threshold_zero_matches_all_nonzero_bots,
        "threshold=1.0 → nobody"       : test_threshold_one_matches_nobody,
        "lower threshold ≥ matches"    : test_lower_threshold_yields_more_or_equal_matches,
        # 4 edge cases
        "empty string"                 : lambda: test_edge_case_does_not_crash(""),
        "very long post"               : lambda: test_edge_case_does_not_crash("bitcoin " * 500),
        "special chars / emoji"        : lambda: test_edge_case_does_not_crash("🚀 OpenAI!!! #AGI $$$"),
        "single short word"            : lambda: test_edge_case_does_not_crash("AI"),
    }

    passed = failed = 0
    print("\n" + "═" * 55)
    print("  Phase 1 — Router Tests")
    print("═" * 55)

    for name, fn in suites.items():
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {name}  ← {e}")
            failed += 1

    print("═" * 55)
    print(f"  {passed} passed  |  {failed} failed  |  {passed + failed} total")
    print("═" * 55 + "\n")
    sys.exit(1 if failed else 0)