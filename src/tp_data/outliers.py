import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def detecter_outliers_iqr(
    df: pd.DataFrame, colonne: str, facteur: float = 1.5
) -> tuple[float, float, int]:
    """Return (borne_basse, borne_haute, nombre_outliers) for a numeric column."""
    serie = df[colonne].dropna()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    borne_basse = float(q1 - facteur * iqr)
    borne_haute = float(q3 + facteur * iqr)
    nombre_outliers = int(((serie < borne_basse) | (serie > borne_haute)).sum())

    print(
        f"{colonne} — bornes normales : [{borne_basse:.1f}, {borne_haute:.1f}]"
        f", outliers détectés : {nombre_outliers}"
    )
    return borne_basse, borne_haute, nombre_outliers


def visualiser_outliers(df: pd.DataFrame, colonnes: list[str]) -> None:
    """Draw a boxplot for each specified column."""
    for col in colonnes:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"{col} : distribution et outliers")
        plt.tight_layout()
        plt.show()
