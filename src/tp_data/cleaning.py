import pandas as pd


def reparer_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert TotalCharges to numeric, impute hidden NaNs with median.

    Raises ValueError if the column is more than 50% non-numeric, indicating
    something other than isolated whitespace entries corrupted the data.
    """
    df = df.copy()

    non_numeric_rate = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().mean()
    if non_numeric_rate > 0.5:
        raise ValueError(
            f"TotalCharges is {non_numeric_rate:.0%} non-numeric — refusing conversion."
        )

    before = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    hidden_nulls = int(df["TotalCharges"].isna().sum() - before)

    print(f"Trous cachés démasqués dans TotalCharges : {hidden_nulls}")

    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    return df, hidden_nulls
