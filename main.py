import pandas as pd

from tp_data.audit import audit_qualite
from tp_data.cleaning import reparer_total_charges
from tp_data.correlations import rapport_multicolinearite
from tp_data.encoding import encoder_features
from tp_data.features import features_discriminantes
from tp_data.outliers import detecter_outliers_iqr
from tp_data.pipeline import bilan_nettoyage, split_et_scale_proprement
from tp_ia.data.dataset import Dataset
from tp_ia.training import DEFAULT_MODELS, Experiment

TELCO_CSV = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    dataset = Dataset("breast_cancer")
    print(dataset)
    print()

    experiment = Experiment(dataset, DEFAULT_MODELS)
    experiment.run()
    print(experiment)
    print()

    df_raw = pd.read_csv(TELCO_CSV)

    audit_qualite(df_raw)
    print()

    df, n_hidden = reparer_total_charges(df_raw)
    print()

    df_enc = encoder_features(df)
    print()

    for col in NUMERIC_COLS:
        detecter_outliers_iqr(df_enc, col)
    print()

    rapport_multicolinearite(df_enc, NUMERIC_COLS)
    print()

    features_discriminantes(df_enc, cible="Churn")
    print()

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]
    X_train, X_test, _, _ = split_et_scale_proprement(X, y)
    print(f"Train : {X_train.shape}  Test : {X_test.shape}")
    print()

    bilan_nettoyage({
        "Colonnes": (df_raw.shape[1], df_enc.shape[1]),
        "Lignes": (df_raw.shape[0], df_enc.shape[0]),
        "Trous cachés": (n_hidden, 0),
    })


if __name__ == "__main__":
    main()
