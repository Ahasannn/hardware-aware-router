"""
load_patterns.py
Implements realistic request arrival patterns for LLM load simulation.

Patterns supported:
  - poisson     : baseline random arrivals
  - microburst  : short high-intensity bursts
  - sustained   : persistent overload
"""

import numpy as np
import time

class RequestPattern:
    def __init__(self, pattern="poisson", rate=5.0, spike_intensity=8.0, spike_period=60.0):
        """
        Args:
            pattern: str ∈ {"poisson", "microburst", "sustained"}
            rate: average arrival rate λ (requests/sec)
            spike_intensity: multiplier during burst period
            spike_period: seconds between bursts
        """
        self.pattern = pattern.lower()
        self.rate = rate
        self.spike_intensity = spike_intensity
        self.spike_period = spike_period
        self._t0 = time.time()  # for burst timing reference

    def next_delay(self) -> float:
        """
        Return the inter-arrival delay (seconds) before next request.
        """
        t = time.time() - self._t0
        base_rate = self.rate

        if self.pattern == "poisson":
            return np.random.exponential(1 / base_rate)

        elif self.pattern == "microburst":
            # every spike_period seconds → 5s burst
            if int(t) % int(self.spike_period) < 5:
                rate = base_rate * self.spike_intensity
            else:
                rate = base_rate
            return np.random.exponential(1 / rate)

        elif self.pattern == "sustained":
            # constant overload (2× rate)
            rate = base_rate * 2
            return np.random.exponential(1 / rate)

        else:
            # fallback to Poisson
            return np.random.exponential(1 / base_rate)
