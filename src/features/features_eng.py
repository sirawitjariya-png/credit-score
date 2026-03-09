import pandas as pd
import numpy as np
import optbinning
from optbinning import BinningProcess

############################
# IV FUNCTIONS
############################

def iv_score(X: pd.DataFrame, y: pd.Series,
             max_n_bins: int = 5,
             min_bin_size: float = 0.05,
             min_n_bins: int = 1,
             param_dict: dict = None):
    """
    Calculate Information Value (IV) using optimal binning.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable (binary)
    max_n_bins : int
        Maximum number of bins
    min_bin_size : float
        Minimum bin size proportion
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    binning_process : BinningProcess
        Fitted binning process object
    iv_summary : pd.DataFrame
        Sorted IV table
    """

    categorical_cols = list(X.select_dtypes(include="object").columns)

    binning_process = BinningProcess(
        variable_names=list(X.columns),
        categorical_variables=categorical_cols,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        min_n_bins=min_n_bins,
        binning_fit_params=param_dict
    )

    binning_process.fit(X, y)

    iv_summary = binning_process.summary()
    iv_summary = iv_summary.sort_values(by="iv", ascending=False)

    return binning_process, iv_summary

############################
# REMOVE HIGH CORRELATION FUNCTION
############################

def remove_correlated_features(
    X: pd.DataFrame,
    iv_df: pd.DataFrame,
    corr_threshold: float = 0.7
):
    """
    Remove highly correlated variables based on IV priority.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    iv_df : pd.DataFrame
        Dataframe with columns ['feature', 'iv']
    corr_threshold : float
        Correlation threshold to consider removal

    Returns
    -------
    selected_features : list
        Remaining features
    removed_features : list
        Removed features due to correlation
    """

    # Sort features by IV (highest first)
    iv_sorted = iv_df.sort_values("iv", ascending=False)
    features = iv_sorted["name"].tolist()

    corr_matrix = X[features].corr().abs()

    selected = []
    removed = []

    for feature in features:

        if feature in removed:
            continue

        selected.append(feature)

        # Find correlated variables
        corr_features = corr_matrix.index[
            (corr_matrix[feature] > corr_threshold) &
            (corr_matrix.index != feature)
        ].tolist()

        for cf in corr_features:
            if cf not in selected:
                removed.append(cf)

    return selected, removed