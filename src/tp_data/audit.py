import pandas as pd


def audit_qualite(df: pd.DataFrame) -> None:
    """Print a health report: shape, missing % per column, Churn distribution."""
    print(f"Forme : {df.shape}")

    missing = df.isna().sum()
    pct = (df.isna().mean() * 100).round(1)
    rapport = pd.DataFrame({"manquants": missing, "pourcent": pct})
    cols_avec_trous = rapport[rapport["manquants"] > 0].sort_values("pourcent", ascending=False)

    if cols_avec_trous.empty:
        print("Manquants détectés : 0 colonne  (méfiance : des trous sont peut-être cachés, voir Phase 2)")
    else:
        print(f"Manquants détectés : {len(cols_avec_trous)} colonne(s)")
        print(cols_avec_trous.to_string())

    if "Churn" in df.columns:
        counts = df["Churn"].value_counts()
        total = len(df)
        for val, count in counts.items():
            print(f"Churn  {val} : {count} ({count / total * 100:.1f}%)")
