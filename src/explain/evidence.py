"""Evidence chain generator.

Produces structured, traceable evidence for each credit decision.
"""

from typing import Dict, List


class EvidenceChainGenerator:
    """Generates evidence chains for credit decisions."""

    def generate_risk_evidence(self, shap_values, features: Dict, top_k: int = 5) -> List[Dict]:
        """Generate risk factor evidence chain."""
        ...

    def generate_causal_evidence(self, ate_results: Dict, cate_value: float,
                                  subgroup: str = None) -> List[Dict]:
        """Generate causal evidence chain."""
        ...

    def generate_counterfactual_evidence(self, cf_result: Dict) -> str:
        """Generate counterfactual evidence text."""
        ...

    def generate_full_evidence_report(self, risk_evidence, causal_evidence,
                                       cf_evidence) -> str:
        """Generate complete evidence report in Markdown."""
        ...
