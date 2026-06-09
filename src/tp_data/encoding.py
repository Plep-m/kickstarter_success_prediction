import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# Contract has a natural order: shortest → longest commitment
_CONTRACT_ORDER = [["Month-to-month", "One year", "Two year"]]

_BINARY_YES_NO = {"Yes": 1, "No": 0}
_BINARY_GENDER = {"Male": 1, "Female": 0}


def encoder_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all categorical columns; return a fully numeric DataFrame.

    Strategy:
    - customerID is dropped (identifier, not a feature).
    - Pure binary Yes/No columns → 0/1.
    - gender (Male/Female) → 0/1.
    - Contract is treated as ordinal (Month-to-month=0, One year=1, Two year=2)
      because a clear duration order exists.
    - All remaining object columns (PaymentMethod, InternetService, etc.) → One-Hot.
    """
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")

    obj_cols = df.select_dtypes("object").columns.tolist()

    for col in obj_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {"Yes", "No"}:
            df[col] = df[col].map(_BINARY_YES_NO)
        elif unique_vals <= {"Male", "Female"}:
            df[col] = df[col].map(_BINARY_GENDER)

    if "Contract" in df.columns and df["Contract"].dtype == object:
        enc = OrdinalEncoder(categories=_CONTRACT_ORDER)
        df["Contract"] = enc.fit_transform(df[["Contract"]])

    remaining_obj = df.select_dtypes("object").columns.tolist()
    if remaining_obj:
        df = pd.get_dummies(df, columns=remaining_obj, prefix=remaining_obj)

    # Ensure all bool columns produced by get_dummies are cast to int
    bool_cols = df.select_dtypes(bool).columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"Colonnes après encodage : {df.shape[1]}")
    return df
