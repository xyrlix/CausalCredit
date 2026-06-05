"""Home Credit Default Risk dataset loader.

Loads `application_train.csv` (or .parquet) from the Home Credit Default Risk
Kaggle competition. Single-table primary dataset for CausalCredit.

Reference columns documented in `docs/CausalCredit_数据集可用性验证分析.md`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS: List[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

NUMERICAL_COLUMNS: List[str] = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE",
    "COMMONAREA_MODE",
    "ELEVATORS_MODE",
    "ENTRANCES_MODE",
    "FLOORSMAX_MODE",
    "FLOORSMIN_MODE",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI",
    "ELEVATORS_MEDI",
    "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_MEDI",
    "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AMT_CREDIT_SUM",
]

# 20 FLAG_DOCUMENT_* are not in numerical list (all stored as int64, treated as binary)
FLAG_DOCUMENT_COLUMNS: List[str] = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]


class HomeCreditLoader:
    """Loader for the Home Credit Default Risk application_train table."""

    def __init__(self, data_dir: str = "data/home-credit-default-risk/"):
        self.data_dir = Path(data_dir)
        self._raw: Optional[pd.DataFrame] = None

    def _resolve_file(self) -> Path:
        for name in ("application_train.parquet", "application_train.csv"):
            path = self.data_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find application_train.{{parquet,csv}} in {self.data_dir}. "
            f"Download from https://www.kaggle.com/c/home-credit-default-risk/data"
        )

    def fetch(self) -> pd.DataFrame:
        """Load and return the Home Credit application_train table."""
        if self._raw is not None:
            return self._raw.copy()
        path = self._resolve_file()
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df = self._fix_known_issues(df)
        self._raw = df
        return df.copy()

    def get_feature_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return features (X) and binary target (y)."""
        df = self.fetch()
        if "TARGET" not in df.columns:
            raise ValueError("TARGET column not found in Home Credit data")
        y = df["TARGET"].astype(int)
        drop_cols = ["TARGET", "SK_ID_CURR"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return X, y

    def get_metadata(self) -> Dict:
        df = self.fetch()
        return {
            "n_samples": len(df),
            "n_features": len(df.columns) - 2,
            "target_distribution": df["TARGET"].value_counts().to_dict(),
            "target_default_rate": float(df["TARGET"].mean()),
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numerical_columns": NUMERICAL_COLUMNS,
            "flag_document_columns": FLAG_DOCUMENT_COLUMNS,
        }

    @staticmethod
    def _fix_known_issues(df: pd.DataFrame) -> pd.DataFrame:
        """Fix known data quality issues documented in dataset analysis.

        - `DAYS_EMPLOYED == 365243` is a sentinel for unemployed (per docs).
        - `CODE_GENDER` contains rare 'XNA' value.
        - `NAME_FAMILY_STATUS` contains rare 'Unknown' value.
        """
        df = df.copy()
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        if "CODE_GENDER" in df.columns:
            df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan
        if "NAME_FAMILY_STATUS" in df.columns:
            df.loc[df["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = np.nan
        if "ORGANIZATION_TYPE" in df.columns:
            df.loc[df["ORGANIZATION_TYPE"] == "XNA", "ORGANIZATION_TYPE"] = np.nan
        return df
