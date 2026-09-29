from __future__ import annotations

import pytest

from evalbench.scorers import REGISTRY, load_scorer
from evalbench.scorers.exact import exact
from evalbench.scorers.field_match import field_match
from evalbench.scorers.numeric import numeric_tolerance


class TestExact:
    def test_identical_dicts_score_one(self):
        assert exact({"a": 1}, {"a": 1}) == 1.0

    def test_key_order_does_not_matter(self):
        assert exact({"a": 1, "b": 2}, {"b": 2, "a": 1}) == 1.0

    def test_string_case_and_whitespace_are_normalised(self):
        assert exact({"c": "INR"}, {"c": " inr "}) == 1.0

    def test_a_different_value_scores_zero(self):
        assert exact({"a": 1}, {"a": 2}) == 0.0

    def test_a_missing_key_scores_zero(self):
        assert exact({"a": 1, "b": 2}, {"a": 1}) == 0.0

    def test_an_extra_key_scores_zero(self):
        assert exact({"a": 1}, {"a": 1, "b": 2}) == 0.0

    def test_a_string_is_not_coerced_to_a_number(self):
        # A model returning "4200" where the schema wants 4200 is wrong, and
        # forgiving it here just moves the breakage somewhere less visible.
        assert exact({"amount": 4200}, {"amount": "4200"}) == 0.0

    def test_nested_structures_are_compared(self):
        assert exact({"a": [1, {"b": "X"}]}, {"a": [1, {"b": "x"}]}) == 1.0


class TestFieldMatch:
    def test_returns_an_aggregate_and_a_breakdown(self):
        score, fields = field_match(
            {"invoice_no": "INV-1", "amount": 4200, "currency": "INR"},
            {"invoice_no": "INV-1", "amount": 4200, "currency": "USD"},
        )
        assert score == pytest.approx(2 / 3)
        assert fields == {"invoice_no": 1.0, "amount": 1.0, "currency": 0.0}

    def test_a_perfect_match_scores_one_on_every_field(self):
        score, fields = field_match({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert score == 1.0
        assert set(fields.values()) == {1.0}

    def test_a_missing_key_scores_zero_for_that_field(self):
        score, fields = field_match({"a": 1, "b": 2}, {"a": 1})
        assert score == 0.5
        assert fields["b"] == 0.0

    def test_an_invented_key_is_reported_but_does_not_dilute_the_score(self):
        # Adding a field and getting one wrong are different problems, and
        # averaging them together hides both.
        score, fields = field_match({"a": 1}, {"a": 1, "surprise": 9})
        assert score == 1.0
        assert fields["surprise.unexpected"] == 0.0

    def test_a_non_dict_response_scores_zero_on_every_expected_field(self):
        score, fields = field_match({"a": 1, "b": 2}, "not a dict")
        assert score == 0.0
        assert fields == {"a": 0.0, "b": 0.0}

    def test_a_non_dict_expectation_falls_back_to_equality(self):
        assert field_match([1, 2], [1, 2]) == (1.0, {})
        assert field_match([1, 2], [1, 3]) == (0.0, {})

    def test_the_breakdown_is_what_isolates_a_failing_category(self):
        # Three invoices where only the currency is ever wrong. The aggregate
        # says 0.67; the breakdown says "currency", which is actionable.
        cases = [
            ({"no": "1", "cur": "INR"}, {"no": "1", "cur": "USD"}),
            ({"no": "2", "cur": "EUR"}, {"no": "2", "cur": "USD"}),
        ]
        breakdowns = [field_match(e, a)[1] for e, a in cases]
        assert all(b["no"] == 1.0 for b in breakdowns)
        assert all(b["cur"] == 0.0 for b in breakdowns)


class TestNumericTolerance:
    def test_exact_equality(self):
        assert numeric_tolerance(100, 100) == 1.0

    def test_within_the_default_tolerance(self):
        assert numeric_tolerance(100, 100.5) == 1.0

    def test_outside_the_default_tolerance(self):
        assert numeric_tolerance(100, 105) == 0.0

    def test_a_custom_tolerance(self):
        assert numeric_tolerance(100, 105, tolerance=0.10) == 1.0

    def test_zero_expected_requires_exact_equality(self):
        assert numeric_tolerance(0, 0) == 1.0
        assert numeric_tolerance(0, 0.0001) == 0.0

    def test_negative_values(self):
        assert numeric_tolerance(-100, -100.5) == 1.0

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [(1, True), (True, 1), (0, False), (1, "1"), (1, None), (1, [1])],
    )
    def test_non_numbers_and_booleans_score_zero(self, expected, actual):
        # Python says True == 1. A scorer that agrees reports a model as
        # correct when it returned the wrong type entirely.
        assert numeric_tolerance(expected, actual) == 0.0


class TestRegistryAndLoading:
    def test_the_registry_names_every_built_in(self):
        assert set(REGISTRY) == {"exact", "field_match", "numeric_tolerance"}

    def test_a_registry_name_resolves(self):
        name, fn = load_scorer("field_match")
        assert name == "field_match"
        assert fn is field_match

    def test_an_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match="available: exact, field_match"):
            load_scorer("nope")

    def test_a_custom_scorer_loads_from_a_path(self, tmp_path):
        module = tmp_path / "my_scorer.py"
        module.write_text(
            "def score(expected, actual):\n"
            "    return 1.0 if str(expected) in str(actual) else 0.0\n",
            encoding="utf-8",
        )
        name, fn = load_scorer(f"{module}:score")
        assert name.endswith(":score")
        assert fn("abc", "xxabcxx") == 1.0
        assert fn("abc", "zzz") == 0.0

    def test_a_missing_module_is_reported_clearly(self):
        with pytest.raises(ValueError, match="scorer module not found"):
            load_scorer("/nonexistent/path.py:score")

    def test_a_missing_function_is_reported_clearly(self, tmp_path):
        module = tmp_path / "empty.py"
        module.write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no callable named"):
            load_scorer(f"{module}:score")
