"""Causal effect refutation and sensitivity analysis.

Implements DoWhy refutation methods to validate ATE estimates:
- Random common cause (placebo test)
- Placebo treatment refuter
- Data subset refuter
- Unobserved confounding sensitivity
"""

from typing import Any, Dict, List


class CausalRefuter:
    """Refutation tester for causal effect estimates."""

    def __init__(self, causal_model: Any):
        self.model = causal_model

    def refute_random_cause(self, estimate, num_simulations: int = 100) -> Dict:
        """Add a random common cause variable and re-estimate ATE."""
        ...

    def refute_placebo_treatment(self, estimate, num_simulations: int = 100) -> Dict:
        """Replace treatment with random placebo and re-estimate."""
        ...

    def refute_data_subset(self, estimate, subset_fraction: float = 0.8,
                           num_simulations: int = 100) -> Dict:
        """Re-estimate ATE on random data subsets to check stability."""
        ...

    def refute_unobserved_confounding(self, estimate) -> Dict:
        """Sensitivity analysis for unobserved confounding."""
        ...

    def run_all_refutations(self, estimate, methods: List[str] | None = None) -> Dict[str, Dict]:
        """Run all specified refutation methods.

        Args:
            estimate: DoWhy causal estimate object.
            methods: List of refutation method names. Defaults to all four.

        Returns:
            Dict mapping method_name -> {original_estimate, refuted_estimate, p_value, is_robust}.
        """
        ...

    def compute_robustness_score(self, refutation_results: Dict[str, Dict]) -> float:
        """Compute overall robustness score (0-1) from refutation results."""
        ...

    def compute_e_value(self, estimate, outcome_range: tuple[float, float]) -> float:
        """Compute E-value for unmeasured confounding sensitivity."""
        ...
