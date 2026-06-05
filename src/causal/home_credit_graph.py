"""Hand-coded causal DAG for the Home Credit Default Risk dataset.

Mirrors the design pattern of `src/causal/graph.py` (German Credit graph).
Based on domain knowledge documented in:
- docs/CausalCredit_完整实现计划书.md section 4 (技术亮点)
- docs/CausalCredit_数据集可用性验证分析.md section 2.1 (特征验证)

Treatments: AMT_CREDIT, AMT_ANNUITY, DAYS_EMPLOYED
Outcome:     TARGET
Confounders: AMT_INCOME_TOTAL, NAME_EDUCATION_TYPE, OCCUPATION_TYPE,
             REGION_RATING_CLIENT, DAYS_BIRTH, CNT_CHILDREN,
             NAME_FAMILY_STATUS, EXT_SOURCE_2
Mediators:   AMT_GOODS_PRICE, NAME_HOUSING_TYPE
Sensitive:   CODE_GENDER
"""

from typing import Dict, List, Set, Tuple


class HomeCreditCausalGraph:
    """Hand-coded causal DAG for the Home Credit Default Risk dataset."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
        self._define_graph()

    def _define_graph(self):
        self.nodes = {
            # Treatments
            "AMT_CREDIT": {"type": "treatment", "label": "Credit Amount"},
            "AMT_ANNUITY": {"type": "treatment", "label": "Annuity"},
            "DAYS_EMPLOYED": {"type": "treatment", "label": "Days Employed"},
            # Outcome
            "TARGET": {"type": "outcome", "label": "Default"},
            # Confounders
            "AMT_INCOME_TOTAL": {"type": "confounder", "label": "Total Income"},
            "NAME_EDUCATION_TYPE": {"type": "confounder", "label": "Education"},
            "OCCUPATION_TYPE": {"type": "confounder", "label": "Occupation"},
            "REGION_RATING_CLIENT": {"type": "confounder", "label": "Region Rating"},
            "DAYS_BIRTH": {"type": "confounder", "label": "Age (days from bday)"},
            "CNT_CHILDREN": {"type": "confounder", "label": "Num Children"},
            "NAME_FAMILY_STATUS": {"type": "confounder", "label": "Family Status"},
            "EXT_SOURCE_2": {"type": "confounder", "label": "EXT Source 2"},
            # Mediators
            "AMT_GOODS_PRICE": {"type": "mediator", "label": "Goods Price"},
            "NAME_HOUSING_TYPE": {"type": "mediator", "label": "Housing Type"},
            # Sensitive attribute
            "CODE_GENDER": {"type": "sensitive", "label": "Gender"},
        }

        self.edges = [
            # Income affects both treatments and outcome
            ("AMT_INCOME_TOTAL", "AMT_CREDIT"),
            ("AMT_INCOME_TOTAL", "AMT_ANNUITY"),
            ("AMT_INCOME_TOTAL", "DAYS_EMPLOYED"),
            ("AMT_INCOME_TOTAL", "TARGET"),
            # Education -> income, employment, default risk
            ("NAME_EDUCATION_TYPE", "AMT_INCOME_TOTAL"),
            ("NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"),
            ("NAME_EDUCATION_TYPE", "DAYS_EMPLOYED"),
            ("NAME_EDUCATION_TYPE", "TARGET"),
            # Occupation -> income, default
            ("OCCUPATION_TYPE", "AMT_INCOME_TOTAL"),
            ("OCCUPATION_TYPE", "DAYS_EMPLOYED"),
            ("OCCUPATION_TYPE", "TARGET"),
            # Region rating -> all treatments + default
            ("REGION_RATING_CLIENT", "AMT_CREDIT"),
            ("REGION_RATING_CLIENT", "AMT_ANNUITY"),
            ("REGION_RATING_CLIENT", "TARGET"),
            # Age -> children, family, employment, default
            ("DAYS_BIRTH", "CNT_CHILDREN"),
            ("DAYS_BIRTH", "NAME_FAMILY_STATUS"),
            ("DAYS_BIRTH", "DAYS_EMPLOYED"),
            ("DAYS_BIRTH", "TARGET"),
            # Children -> family status, default
            ("CNT_CHILDREN", "NAME_FAMILY_STATUS"),
            ("CNT_CHILDREN", "TARGET"),
            # Family status -> housing, default
            ("NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"),
            ("NAME_FAMILY_STATUS", "TARGET"),
            # EXT_SOURCE_2 -> default (external risk score)
            ("EXT_SOURCE_2", "TARGET"),
            # Gender -> employment, default (for fairness check)
            ("CODE_GENDER", "DAYS_EMPLOYED"),
            ("CODE_GENDER", "OCCUPATION_TYPE"),
            ("CODE_GENDER", "AMT_INCOME_TOTAL"),
            # Treatment -> mediator
            ("AMT_CREDIT", "AMT_GOODS_PRICE"),
            ("AMT_ANNUITY", "AMT_GOODS_PRICE"),
            # Treatment -> outcome (direct)
            ("AMT_CREDIT", "TARGET"),
            ("AMT_ANNUITY", "TARGET"),
            ("DAYS_EMPLOYED", "TARGET"),
            # Mediator -> outcome
            ("AMT_GOODS_PRICE", "TARGET"),
            ("NAME_HOUSING_TYPE", "TARGET"),
        ]

    # ------------------------------------------------------------------ accessors
    def get_treatment_variables(self) -> List[str]:
        return [n for n, p in self.nodes.items() if p.get("type") == "treatment"]

    def get_outcome_variable(self) -> str:
        outcomes = [n for n, p in self.nodes.items() if p.get("type") == "outcome"]
        return outcomes[0] if outcomes else "TARGET"

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Return confounders: ancestors of both treatment and outcome."""
        ancestors_tx = self._get_ancestors(treatment)
        ancestors_out = self._get_ancestors(outcome)
        return [
            n for n, p in self.nodes.items()
            if p.get("type") in ("confounder", "sensitive")
            and n in ancestors_tx and n in ancestors_out
            and n not in (treatment, outcome)
        ]

    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """Return mediators on the directed path from treatment to outcome."""
        return [
            n for n, p in self.nodes.items()
            if p.get("type") == "mediator"
            and n not in (treatment, outcome)
            and self._is_on_path(treatment, outcome, n)
        ]

    def get_instruments(self, treatment: str) -> List[str]:
        """Return variables that affect treatment but not the outcome directly.

        An instrument X satisfies: X -> treatment, and no direct edge X -> outcome.
        """
        outcome = self.get_outcome_variable()
        instruments = []
        for n, p in self.nodes.items():
            if n == treatment:
                continue
            affects_tx = self._is_ancestor(n, treatment)
            direct_to_outcome = (n, outcome) in self.edges
            if affects_tx and not direct_to_outcome:
                instruments.append(n)
        return instruments

    def get_sensitive_attributes(self) -> List[str]:
        return [n for n, p in self.nodes.items() if p.get("type") == "sensitive"]

    def validate_acyclic(self) -> bool:
        """DFS-based cycle detection."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for src, dst in self.edges:
                if src == node:
                    if dst not in visited:
                        if dfs(dst):
                            return True
                    elif dst in rec_stack:
                        return True
            rec_stack.discard(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return False
        return True

    def get_assumptions(self) -> List[str]:
        return [
            "Unconfoundedness: All common causes of treatment and outcome are observed in the DAG.",
            "Positivity: Every applicant has non-zero probability of receiving each treatment level (relevant for continuous treatments).",
            "SUTVA: The outcome of one applicant is unaffected by the treatment of other applicants.",
            "Consistency: Observed TARGET equals the potential outcome under observed treatment.",
            "No measurement error: All variables are observed without error (approximate — missing values are imputed).",
            "Correct DAG structure: The hand-coded edges reflect the true causal structure for Home Credit.",
        ]

    def get_dot_string(self) -> str:
        lines = ['digraph HomeCredit {', '  rankdir="TB";', '  node [shape=ellipse, style=filled];']
        colors = {
            "treatment": "#FFB3BA",
            "outcome": "#BAFFC9",
            "confounder": "#BAE1FF",
            "mediator": "#FFFFBA",
            "sensitive": "#FFDFBA",
        }
        for node_id, props in self.nodes.items():
            node_type = props.get("type", "unknown")
            color = colors.get(node_type, "#FFFFFF")
            label = props.get("label", node_id)
            lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')
        for src, dst in self.edges:
            lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------- internal helpers
    def _get_ancestors(self, node: str) -> Set[str]:
        ancestors: Set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            for src, dst in self.edges:
                if dst == current and src not in ancestors:
                    ancestors.add(src)
                    queue.append(src)
        return ancestors

    def _get_descendants(self, node: str) -> Set[str]:
        descendants: Set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            for src, dst in self.edges:
                if src == current and dst not in descendants:
                    descendants.add(dst)
                    queue.append(dst)
        return descendants

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return descendant in self._get_descendants(ancestor)

    def _is_on_path(self, source: str, target: str, middle: str) -> bool:
        if middle in (source, target):
            return False
        return self._is_ancestor(source, middle) and self._is_ancestor(middle, target)
