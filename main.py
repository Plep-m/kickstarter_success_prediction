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
from tp_ia_2_j3.phase_a_regression import HousingBenchmark
from tp_ia_2_j3.phase_b_clustering import AirbnbSegmenter
from tp_ia_2_j3.phase_c_spam import SpamBenchmark
from tp_ia_2_j3.phase_d_sonar import SONAR_URL, SonarBenchmark
from tp_ia_2_j3.phase_e_fight import IAFight, Leaderboard, f1_binary
from tp_ia_2_j3.phase_d_sonar import SonarLoader
from sklearn.model_selection import train_test_split

TELCO_CSV = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

AIRBNB_SOURCE = "data/listings.csv"
SPAM_PATH = "data/SMSSpamCollection"

_SECTION = "=" * 60


def _header(title: str) -> None:
    print(f"\n{_SECTION}\n  {title}\n{_SECTION}")


def showcase_j1_j2() -> None:
    _header("J1/J2 — Dataset & Telco pipeline")

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


def showcase_j3() -> None:
    _header("J3 — Phase A : Régression immobilière (California Housing)")
    HousingBenchmark().run().report()

    _header("J3 — Phase B : Clustering AirBnB (non supervisé)")
    try:
        AirbnbSegmenter(AIRBNB_SOURCE).run().report()
    except FileNotFoundError:
        print(f"  Données absentes — déposez un listings.csv dans {AIRBNB_SOURCE}")
    except Exception as exc:
        print(f"  Phase B ignorée ({exc.__class__.__name__}: {exc})")

    _header("J3 — Phase C : Spam SMS (NaiveBayes vs LogisticRegression)")
    try:
        SpamBenchmark(SPAM_PATH).run().report()
    except FileNotFoundError:
        print(f"  Données absentes — déposez le SMS Spam Collection UCI dans {SPAM_PATH}")
    except Exception as exc:
        print(f"  Phase C ignorée ({exc.__class__.__name__}: {exc})")

    _header("J3 — Phase D : Sonar — classification binaire mine vs rocher")
    SonarBenchmark(SONAR_URL).run().report()

    _header("J3 — Phase E : Fight des IA — leaderboard sonar")
    X, y = SonarLoader(SONAR_URL).load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    fight = IAFight(X_train, X_test, y_train, y_test, metric_fn=f1_binary, scale=True)
    fight.run()
    board = Leaderboard(fight.results, dataset_name="sonar", metric_name="F1")
    print(board)
    print(f"\nChampion : {fight.champion().name}")


def main() -> None:
    showcase_j1_j2()
    showcase_j3()


if __name__ == "__main__":
    main()
