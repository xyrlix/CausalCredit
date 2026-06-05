"""Credit default causal DAG definition and validation.

Defines the causal directed acyclic graph for credit default analysis
based on domain knowledge. The graph defines relationships between:

Treatments: credit_amount, duration
Outcome: default (class)
Confounders: age, job, housing, savings_status, checking_status, employment
Mediators: installment_commitment
"""

from typing import Dict, List, Set, Tuple


class CreditCausalGraph:
    """Causal graph for credit default analysis."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
        self._define_graph()

    def _define_graph(self):
        """Define the causal graph structure based on domain knowledge."""
        self.nodes = {
            "credit_amount": {"type": "treatment", "label": "Credit Amount"},
            "duration": {"type": "treatment", "label": "Loan Duration"},
            "class": {"type": "outcome", "label": "Default"},
            "age": {"type": "confounder", "label": "Age"},
            "job": {"type": "confounder", "label": "Job Type"},
            "housing": {"type": "confounder", "label": "Housing Status"},
            "savings_status": {"type": "confounder", "label": "Savings Status"},
            "checking_status": {"type": "confounder", "label": "Checking Status"},
            "employment": {"type": "confounder", "label": "Employment Status"},
            "credit_history": {"type": "confounder", "label": "Credit History"},
            "purpose": {"type": "confounder", "label": "Loan Purpose"},
            "installment_commitment": {"type": "mediator", "label": "Installment Rate"},
        }

        self.edges = [
            ("age", "credit_amount"),
            ("age", "duration"),
            ("age", "class"),
            ("job", "credit_amount"),
            ("job", "duration"),
            ("job", "class"),
            ("housing", "credit_amount"),
            ("housing", "class"),
            ("savings_status", "credit_amount"),
            ("savings_status", "class"),
            ("checking_status", "credit_amount"),
            ("checking_status", "duration"),
            ("checking_status", "class"),
            ("employment", "credit_amount"),
            ("employment", "duration"),
            ("employment", "class"),
            ("credit_history", "credit_amount"),
            ("credit_history", "class"),
            ("purpose", "credit_amount"),
            ("purpose", "duration"),
            ("purpose", "class"),
            ("credit_amount", "installment_commitment"),
            ("duration", "installment_commitment"),
            ("credit_amount", "class"),
            ("duration", "class"),
            ("installment_commitment", "class"),
        ]

    def get_dot_string(self) -> str:
        """Return DOT format causal graph string for visualization."""
        lines = ["digraph CausalCredit {"]
        lines.append('  rankdir="TB";')
        lines.append('  node [shape=ellipse, style=filled];')

        for node_id, props in self.nodes.items():
            node_type = props.get("type", "unknown")
            colors = {
                "treatment": "#FFB3BA",
                "outcome": "#BAFFC9",
                "confounder": "#BAE1FF",
                "mediator": "#FFFFBA",
            }
            color = colors.get(node_type, "#FFFFFF")
            label = props.get("label", node_id)
            lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')

        for src, dst in self.edges:
            lines.append(f'  "{src}" -> "{dst}";')

        lines.append("}")
        return "\n".join(lines)

    def get_treatment_variables(self) -> List[str]:
        """Return treatment variable names."""
        return [n for n, p in self.nodes.items() if p.get("type") == "treatment"]

    def get_outcome_variable(self) -> str:
        """Return the outcome variable name."""
        outcomes = [n for n, p in self.nodes.items() if p.get("type") == "outcome"]
        return outcomes[0] if outcomes else "class"

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Return confounder list for a given treatment-outcome pair.

        Confounders are variables that affect both the treatment and the outcome.
        """
        ancestors_tx = self._get_ancestors(treatment)
        ancestors_out = self._get_ancestors(outcome)

        confounders = []
        for node, props in self.nodes.items():
            if node in (treatment, outcome):
                continue
            if props.get("type") in ("confounder",) and node in ancestors_tx and node in ancestors_out:
                confounders.append(node)

        return confounders

    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """Return mediator list for a given treatment-outcome pair.

        Mediators are variables on the causal path from treatment to outcome.
        """
        mediators = []
        for node, props in self.nodes.items():
            if node in (treatment, outcome):
                continue
            if props.get("type") == "mediator":
                if self._is_on_path(treatment, outcome, node):
                    mediators.append(node)
        return mediators

    def get_instruments(self, treatment: str) -> List[str]:
        """Return potential instrument variable candidates.

        Instruments affect treatment but have no direct effect on outcome.
        """
        outcome = self.get_outcome_variable()
        instruments = []
        for node, props in self.nodes.items():
            if node == treatment:
                continue
            affects_tx = self._is_ancestor(node, treatment)
            direct_to_outcome = (node, outcome) in self.edges
            if affects_tx and not direct_to_outcome:
                instruments.append(node)
        return instruments

    def validate_acyclic(self) -> bool:
        """Verify the causal graph has no cycles using DFS."""
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
        """Return the key assumptions underlying the causal graph."""
        return [
            "Unconfoundedness: All common causes of treatment and outcome are observed.",
            "Positivity: Every unit has a non-zero probability of receiving each treatment level.",
            "SUTVA: The outcome of one unit is unaffected by the treatment assignment of other units.",
            "Consistency: The observed outcome equals the potential outcome under the observed treatment.",
            "No measurement error: All variables are measured without error.",
            "Correct functional form: The DAG correctly represents the causal structure.",
        ]

    def _get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestor nodes of a given node."""
        ancestors: Set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            for src, dst in self.edges:
                if dst == current and src not in ancestors:
                    ancestors.add(src)
                    queue.append(src)
        return ancestors

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is truly an ancestor of descendant in the graph."""
        return descendant in self._get_descendants(ancestor)

    def _get_descendants(self, node: str) -> Set[str]:
        """Get all descendant nodes of a given node."""
        descendants: Set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            for src, dst in self.edges:
                if src == current and dst not in descendants:
                    descendants.add(dst)
                    queue.append(dst)
        return descendants

    def _is_on_path(self, source: str, target: str, middle: str) -> bool:
        """Check if middle is on at least one directed path from source to target."""
        if middle == source or middle == target:
            return False
        return self._is_ancestor(source, middle) and self._is_ancestor(middle, target)
