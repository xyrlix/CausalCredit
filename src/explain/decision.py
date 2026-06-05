"""Decision advisory engine.

Generates personalized loan decision recommendations based on
causal analysis and counterfactual reasoning.
"""

from typing import Dict


class DecisionAdvisor:
    """Credit decision advisory engine."""

    def __init__(self, counterfactual_reasoner):
        self.counterfactual_reasoner = counterfactual_reasoner

    def generate_decision_report(self, features: Dict[str, float],
                                  include_shap: bool = True) -> Dict:
        """Generate a complete decision report."""
        ...

    def generate_suggestion_text(self, report: Dict, language: str = "zh") -> str:
        """Generate human-readable decision suggestion text."""
        ...

    def compare_subgroup_effect(self, features: Dict[str, float],
                                 subgroup_col: str = "thin_credit_flag") -> Dict:
        """Compare causal effects across subgroups."""
        ...
