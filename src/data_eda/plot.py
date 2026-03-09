import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############################
# NUMERICAL PLOT FUNCTIONS
############################

def plot_numeric_distribution(df: pd.DataFrame, cols=None, bins=30):
    """
    Plot histogram and boxplot for numeric variables
    """

    num_df = df.select_dtypes(include=np.number)

    if cols is not None:
        num_df = num_df[cols]

    for col in num_df.columns:

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(num_df[col], bins=bins, kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution: {col}")

        sns.boxplot(x=num_df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot: {col}")

        plt.tight_layout()
        plt.show()

def plot_numeric_vs_target_boxplot(df: pd.DataFrame, target: str, cols=None):
    """
    Plot boxplot of numeric variables against target
    """

    num_df = df.select_dtypes(include=np.number)

    if cols is not None:
        num_df = num_df[cols]

    for col in num_df.columns:

        if col == target:
            continue

        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f"{col} vs {target}")
        plt.show()

def plot_numeric_vs_target_hist(df: pd.DataFrame, target: str, cols=None, bins=30):
    """
    Plot histogram of numeric variables split by target class
    """

    num_df = df.select_dtypes(include=np.number)

    if cols is not None:
        num_df = num_df[cols]

    for col in num_df.columns:

        if col == target:
            continue

        plt.figure(figsize=(6,4))

        sns.histplot(
            data=df,
            x=col,
            hue=target,
            bins=bins,
            kde=True,
            stat="density",
            common_norm=False
        )

        plt.title(f"{col} distribution by {target}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.show()

def plot_numeric_vs_target_distribution(df: pd.DataFrame, target: str, cols=None):
    """
    Plot smooth distribution (KDE) of numeric variables by target
    """

    num_df = df.select_dtypes(include=np.number)

    if cols is not None:
        num_df = num_df[cols]

    for col in num_df.columns:

        if col == target:
            continue

        plt.figure(figsize=(6,4))

        sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            fill=True,
            alpha=0.3,     # transparency
            common_norm=False
        )

        plt.title(f"{col} distribution by {target}")
        plt.xlabel(col)
        plt.ylabel("Density")

        plt.show()

############################
# CATEGORICAL PLOT FUNCTIONS
############################

def plot_categorical_distribution(df: pd.DataFrame, cols=None, top_n=20):
    """
    Plot countplot for categorical variables
    """

    cat_df = df.select_dtypes(include="object")

    if cols is not None:
        cat_df = cat_df[cols]

    for col in cat_df.columns:

        plt.figure(figsize=(8,4))

        top_categories = df[col].value_counts().nlargest(top_n).index

        sns.countplot(
            data=df[df[col].isin(top_categories)],
            x=col,
            order=top_categories
        )

        plt.title(f"Category Distribution: {col}")
        plt.xticks(rotation=45)
        plt.show()

def plot_categorical_vs_target(df: pd.DataFrame, target: str, cols=None):
    """
    Plot stacked bar showing target distribution by category
    """

    cat_df = df.select_dtypes(include="object")

    if cols is not None:
        cat_df = cat_df[cols]

    for col in cat_df.columns:

        cross_tab = pd.crosstab(df[col], df[target], normalize="index")

        cross_tab.plot(kind="bar", stacked=True, figsize=(8,4))

        plt.title(f"{col} vs {target}")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)

        plt.show()


############################
# CORRELATION FUNCTIONS
############################

def plot_correlation_matrix(df: pd.DataFrame, method: str = "pearson"):
    """
    Plot correlation matrix with annotation

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    """

    num_df = df.select_dtypes(include=np.number)

    corr = num_df.corr(method=method)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,            # show correlation values
        fmt=".2f",             # 2 decimal format
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 9}
    )

    plt.title(f"{method.capitalize()} Correlation Matrix")
    plt.tight_layout()
    plt.show()