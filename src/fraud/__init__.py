"""Anti-fraud detection module.

Implements the three-pillar fraud detection from
``docs/CausalCredit_反欺诈能力覆盖分析.md`` §4.1:

* ``three_class`` — three-class defaulter sub-classifier
  (fraudulent / non-malicious / systemic)
* ``packaging``  — packaging detection via SHAP×causal four-quadrant
  (TRUSTED / UNTRUSTED / MASKED / NEGLIGIBLE) + path integrity
* ``denoising``  — causal denoising scorer
  ``P(真实评分 | do(去除养流水效应))`` using M5+ temporal features

See ``pipeline.FraudGuard`` for the orchestrator that wires the three
modules into a single ``fraud_score / packaging_score / denoised_score``
triple per applicant.
"""

from src.fraud.three_class import ThreeClassFraudClassifier, DEFRAUDER_CLASSES, ALL_LABELS
from src.fraud.packaging import PackagingDetector
from src.fraud.denoising import CausalDenoisingScorer
from src.fraud.pipeline import FraudGuard

__all__ = [
    "ThreeClassFraudClassifier",
    "DEFRAUDER_CLASSES",
    "ALL_LABELS",
    "PackagingDetector",
    "CausalDenoisingScorer",
    "FraudGuard",
]
