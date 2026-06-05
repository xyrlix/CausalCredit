"""ATE estimation and refutation using DoWhy.

Estimates Average Treatment Effects for interest_rate, credit_amount,
and credit_term on default probability.
"""

from typing import Dict, List

import pandas as pd


class CausalEffectEstimator:
    """Causal effect estimator using DoWhy."""

    def __init__(self, causal_graph, data: pd.DataFrame):
        self.graph = causal_graph
        self.data = data

    def estimate_ate(self, treatment: str, outcome: str, method: str = "backdoor") -> Dict:
        """Estimate Average Treatment Effect."""
        ...

    def estimate_all_treatments(self, outcome: str = "default") -> pd.DataFrame:
        """Estimate ATE for all treatment variables."""
        ...

    def refute_estimate(self, treatment: str, outcome: str,
                        ate_estimate, refutation_methods: List[str] = None) -> Dict:
        """Run refutation tests on ATE estimates."""
        ...

    def comprehensive_analysis(self, treatment: str, outcome: str = "default") -> Dict:
        """Run comprehensive analysis: multiple estimation methods + refutation."""
        ...
