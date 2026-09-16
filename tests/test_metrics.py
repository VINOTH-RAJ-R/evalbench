from __future__ import annotations

import pytest

from evalbench.metrics import consistency, mean, percentile


class TestPercentile:
    """Nearest-rank, inclusive. Every result is an observed value."""

    def test_p50_of_an_odd_sample(self):
        assert percentile([10, 20, 30], 0.50) == 20

    def test_p50_of_an_even_sample_takes_the_upper_of_the_middle_pair(self):
        # Interpolation would return 25, a latency that never occurred.
        assert percentile([10, 20, 30, 40], 0.50) == 20

    def test_p95_returns_an_observed_value(self):
        values = list(range(1, 21))  # 1..20
        result = percentile(values, 0.95)
        assert result in values
        assert result == 19  # ceil(0.95 * 20) - 1 = index 18

    def test_p100_is_the_maximum(self):
        assert percentile([5, 1, 9, 3], 1.0) == 9

    def test_a_single_value_is_its_own_percentile(self):
        for fraction in (0.5, 0.95, 1.0):
            assert percentile([42.0], fraction) == 42.0

    def test_input_order_does_not_matter(self):
        assert percentile([30, 10, 20], 0.5) == percentile([10, 20, 30], 0.5)

    def test_an_empty_sample_is_zero_not_an_error(self):
        assert percentile([], 0.95) == 0.0

    def test_a_known_distribution_with_a_long_tail(self):
        # 19 fast requests and one very slow one: p50 stays low, p95 catches it.
        values = [100.0] * 19 + [5000.0]
        assert percentile(values, 0.50) == 100.0
        assert percentile(values, 0.95) == 100.0
        assert percentile(values, 1.00) == 5000.0

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_a_fraction_outside_the_valid_range_is_rejected(self, fraction: float):
        with pytest.raises(ValueError, match="fraction must be"):
            percentile([1.0, 2.0], fraction)

    def test_every_reported_percentile_appears_in_the_input(self):
        values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0]
        for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0):
            assert percentile(values, fraction) in values


class TestConsistency:
    def test_identical_repeats_are_perfectly_consistent(self):
        assert consistency([1.0, 1.0, 1.0]) == 1.0
        assert consistency([0.0, 0.0]) == 1.0

    def test_maximally_opposed_repeats_score_zero(self):
        assert consistency([0.0, 1.0]) == 0.0

    def test_a_single_repeat_is_undefined_rather_than_perfect(self):
        # Returning 1.0 would claim repeatability was measured when nothing was.
        assert consistency([1.0]) is None
        assert consistency([]) is None

    def test_partial_agreement_is_between_the_extremes(self):
        result = consistency([1.0, 1.0, 0.0])
        assert result is not None
        assert 0.0 < result < 1.0
        # pairs: (1,1)->1.0, (1,0)->0.0, (1,0)->0.0  => mean 1/3
        assert result == pytest.approx(1 / 3)

    def test_continuous_scores_are_handled(self):
        assert consistency([0.8, 0.9]) == pytest.approx(0.9)

    def test_order_does_not_matter(self):
        assert consistency([1.0, 0.5, 0.0]) == consistency([0.0, 1.0, 0.5])

    def test_it_is_not_just_variance_in_disguise(self):
        # Two binary samples with the same mean but different spread must be
        # distinguishable, which is the failure mode of the variance phrasing.
        assert consistency([1.0, 1.0, 0.0, 0.0]) != consistency([0.5, 0.5, 0.5, 0.5])


class TestMean:
    def test_empty_is_zero(self):
        assert mean([]) == 0.0

    def test_ordinary_case(self):
        assert mean([1.0, 2.0, 3.0]) == 2.0
