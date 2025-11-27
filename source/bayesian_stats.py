from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.stats import beta


class BayesianEstimator:
    def __init__(self, outcomes_list: List[str]):
        self.outcome_map = {outcome: i for i, outcome in enumerate(outcomes_list)}
        self.num_outcomes = len(outcomes_list)

        self.prior_alphas = np.ones(self.num_outcomes, dtype=float)

    def calculate_posterior_alphas(self, observed_outcomes: List[str]) -> np.ndarray:
        counts = np.zeros(self.num_outcomes, dtype=float)
        for outcome in observed_outcomes:
            if outcome in self.outcome_map:
                idx = self.outcome_map[outcome]
                counts[idx] += 1.0

        return self.prior_alphas + counts

    def get_credible_interval(
        self,
        posterior_alphas: np.ndarray,
        outcome_key: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        if outcome_key not in self.outcome_map:
            raise ValueError(f"Unknown outcome: {outcome_key}")

        i = self.outcome_map[outcome_key]
        alpha_i = posterior_alphas[i]
        alpha_not_i = np.sum(posterior_alphas) - alpha_i

        lower_p = (1.0 - confidence) / 2.0
        upper_p = 1.0 - lower_p

        lower = beta.ppf(lower_p, a=alpha_i, b=alpha_not_i)
        upper = beta.ppf(upper_p, a=alpha_i, b=alpha_not_i)
        return float(lower), float(upper)
