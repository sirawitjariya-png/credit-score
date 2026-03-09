import pandas as pd
import numpy as np
from optbinning import BinningProcess


############################
# NUMERICAL EDA FUNCTIONS
############################

def num_summary(df: pd.DataFrame):
    """
    Basic statistical summary for numerical variables
    """
    num_df = df.select_dtypes(include=np.number)
    summary = num_df.describe().T
    summary["missing"] = num_df.isnull().sum()
    summary["missing_pct"] = num_df.isnull().mean()
    return summary


def _recommend_transformation(skew, outlier_pct):
    """
    Recommend transformation or scaler based on distribution characteristics
    """

    if skew > 1:
        return "log_transform"

    if outlier_pct > 0.1:
        return "RobustScaler"

    if abs(skew) < 0.5:
        return "StandardScaler"

    return "StandardScaler"


def num_outlier_iqr(df: pd.DataFrame):
    """
    Detect outliers using IQR method, check distribution shape,
    and recommend transformation/scaler.
    """

    num_df = df.select_dtypes(include=np.number)

    outlier_summary = []

    for col in num_df.columns:

        series = num_df[col].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((series < lower) | (series > upper)).sum()

        skew = series.skew()
        kurt = series.kurt()

        outlier_pct = outliers / len(series)

        normal_flag = abs(skew) < 0.5

        recommendation = _recommend_transformation(skew, outlier_pct)

        outlier_summary.append({
            "variable": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "num_outliers": outliers,
            "outlier_pct": outlier_pct,
            "skewness": skew,
            "kurtosis": kurt,
            "normal_dist_flag": normal_flag,
            "recommendation": recommendation
        })

    return pd.DataFrame(outlier_summary)


############################
# CATEGORICAL EDA FUNCTIONS
############################

def cat_summary(df: pd.DataFrame):
    """
    Basic summary for categorical variables
    """
    cat_df = df.select_dtypes(include=["object", "category"])

    summary = pd.DataFrame({
        "variable": cat_df.columns,
        "num_unique": cat_df.nunique(),
        "missing_count": cat_df.isnull().sum(),
        "missing_pct": cat_df.isnull().mean()
    })

    return summary.sort_values("num_unique", ascending=False)


def cat_frequency(df: pd.DataFrame, col: str):
    """
    Frequency table for a categorical variable
    """
    freq = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(normalize=True, dropna=False)

    return pd.DataFrame({
        "count": freq,
        "percentage": pct
    })


def cat_rare_levels(df: pd.DataFrame, threshold=0.01):
    """
    Identify rare categories
    """
    cat_df = df.select_dtypes(include=["object", "category"])

    rare_dict = {}

    for col in cat_df.columns:
        freq = cat_df[col].value_counts(normalize=True)
        rare = freq[freq < threshold]

        if len(rare) > 0:
            rare_dict[col] = rare
    

    return rare_dict



############################
# GENERAL EDA FUNCTIONS
############################

def missing_report(df: pd.DataFrame):
    """
    Missing value report for entire dataset
    """
    report = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isnull().sum(),
        "missing_pct": df.isnull().mean(),
        "type": df.dtypes,
        "unique_value":df.nunique()
    })

    return report.sort_values("missing_pct", ascending=False)