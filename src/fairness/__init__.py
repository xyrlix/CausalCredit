"""Fairness audit module.

Implements the three standard group-fairness metrics for binary
classification as defined in
``docs/CausalCredit_因果推理验证标准体系.md`` §3.2 (recommended by
HKMA / EU AI Act / IEEE 7003 for credit-scoring):

* **Demographic Parity (DP)** — P(Ŷ=1 | A=a) should be the same
  across protected groups ``A``.  Measured as max - min selection
  rate across groups; "< 0.05" is considered fair in BOCHK-style
  guidance.

* **Equal Opportunity (EO)** — TPR (true positive rate) should be
  the same across groups.  Measured as max - min TPR; "< 0.05" is
  fair.  This matters most for credit: a model that over-rejects
  qualified applicants from a group is **legally** a disparate
  treatment.

* **Disparate Impact (DI)** — selection-rate ratio between the
  least-favoured and most-favoured group.  The US "80 % rule"
  (EEOC) requires ``DI >= 0.80``; we flag anything below 0.80 as
  a high-risk disparity.

Slicing attributes (Home Credit flavour):
* ``CODE_GENDER`` — sensitive (legally protected)
* Age bucket derived from ``DAYS_BIRTH`` (young / mid / old)
* Income bucket derived from ``AMT_INCOME_TOTAL`` (low / mid / high)
* Education bucket from ``NAME_EDUCATION_TYPE``

Each slice yields a per-group ``{auc, fpr, fnr, tpr, selection_rate, n}``
dict plus a single overall ``{dp_gap, eo_gap, di_ratio, status}``
summary.  Status is one of ``FAIR`` (both gaps < 0.05 AND DI >= 0.80),
``WARNING`` (one metric just outside) or ``UNFAIR`` (clearly
disparate).
"""

from src.fairness.metrics import (
    demographic_parity_gap,
    equal_opportunity_gap,
    disparate_impact_ratio,
    group_rates,
    summarize_fairness,
    FairnessSummary,
)
from src.fairness.slicing import (
    build_default_slices,
    slice_dataset,
    SLICE_DEFINITIONS,
)
from src.fairness.visualize import (
    plot_group_rates,
    plot_metric_gaps,
    plot_status_board,
    render_all,
)

__all__ = [
    "demographic_parity_gap",
    "equal_opportunity_gap",
    "disparate_impact_ratio",
    "group_rates",
    "summarize_fairness",
    "FairnessSummary",
    "build_default_slices",
    "slice_dataset",
    "SLICE_DEFINITIONS",
    "plot_group_rates",
    "plot_metric_gaps",
    "plot_status_board",
    "render_all",
]
