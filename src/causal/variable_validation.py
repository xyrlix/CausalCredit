"""Causal variable validation.

Validates that causal graph variables are correctly built in the feature
engineering pipeline, with proper distributions and relationships.
"""

from typing import Dict, List

import pandas as pd


class CausalVariableValidator:
    """Validator for causal variables."""

    def validate_treatment_variables(self, df: pd.DataFrame, treatments: List[str]) -> Dict[str, Dict]:
        """Validate treatment variable distributions."""
        ...

    def validate_confounders(self, df: pd.DataFrame, treatments: List[str],
                             outcome: str, confounders: List[str]) -> Dict[str, Dict]:
        """Validate confounder relationships."""
        ...

    def validate_mediators(self, df: pd.DataFrame, treatments: List[str],
                           outcome: str, mediators: List[str]) -> Dict[str, Dict]:
        """Validate mediator pathways."""
        ...

    def validate_instruments(self, df: pd.DataFrame, treatment: str,
                              instruments: List[str]) -> Dict[str, Dict]:
        """Validate instrument variable strength."""
        ...

    def generate_quality_report(self, validation_results: Dict) -> str:
        """Generate causal variable quality report in Markdown."""
        ...
