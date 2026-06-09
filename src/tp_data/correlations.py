import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def rapport_multicolinearite(df: pd.DataFrame, colonnes_num: list[str]) -> pd.DataFrame:
    """Show correlation heatmap and VIF table; return VIF DataFrame."""
    sous_df = df[colonnes_num].dropna()

    matrice = sous_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Corrélations entre variables numériques")
    plt.tight_layout()
    plt.show()

    vif = pd.DataFrame({
        "variable": colonnes_num,
        "VIF": [
            variance_inflation_factor(sous_df.values, i)
            for i in range(len(colonnes_num))
        ],
    }).sort_values("VIF", ascending=False)

    print("\nVIF par variable :")
    print(vif.to_string(index=False))

    high_vif = vif[vif["VIF"] > 5]["variable"].tolist()
    if high_vif:
        print(f"\nVariables au VIF > 5 (multicolinéarité problématique) : {high_vif}")

    return vif
