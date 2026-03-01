"""Unit tests for hw_router.load_patterns."""

from hw_router.load_patterns import RequestPattern


class TestRequestPattern:
    def test_poisson_delay_positive(self):
        pattern = RequestPattern("poisson", rate=5.0)
        for _ in range(20):
            delay = pattern.next_delay()
            assert delay > 0

    def test_microburst_delay_positive(self):
        pattern = RequestPattern("microburst", rate=5.0)
        for _ in range(20):
            delay = pattern.next_delay()
            assert delay > 0

    def test_sustained_delay_positive(self):
        pattern = RequestPattern("sustained", rate=5.0)
        for _ in range(20):
            delay = pattern.next_delay()
            assert delay > 0

    def test_default_parameters(self):
        pattern = RequestPattern()
        delay = pattern.next_delay()
        assert delay > 0

    def test_unknown_pattern_falls_back(self):
        """Unknown patterns fall back to base rate."""
        pattern = RequestPattern("nonexistent", rate=5.0)
        delay = pattern.next_delay()
        assert delay > 0
