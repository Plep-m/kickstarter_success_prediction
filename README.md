# TP IA ML IPPSI 2026

Data exploration project using scikit-learn datasets.<br>
Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

## Tasks

| Command | Description |
|---|---|
| `make lint` | Lint with ruff |
| `make format` | Format with ruff |
| `make ci` | Lint + test |

## Structure

```
src/tp_ia/data/dataset.py     — chargement et découpage des données
src/tp_ia/training/           — expérimentation et comparaison de modèles
tests/                        — suite de tests
```

---

## L'Arène — Comparaison des algorithmes

### Le problème

Deux jeux de données ont été utilisés :

- **Breast Cancer** (569 patients, 30 variables) : prédire si une tumeur est maligne ou bénigne. Une erreur ici a des conséquences réelles — rater un cancer est bien pire que déclencher un faux positif.
- **Wine** (178 échantillons, 13 variables chimiques) : classer un vin parmi 3 cépages. Problème équilibré, enjeu plus académique.

### Tableau de classement

**Breast Cancer**

| Algorithme | Val Acc | Test Acc | Test F1 |
|---|---|---|---|
| Logistic Regression | 1.000 | 0.965 | 0.965 |
| SGD | 1.000 | 0.953 | 0.954 |
| Gradient Boosting | 0.988 | 0.942 | 0.942 |
| XGBoost | 0.988 | 0.953 | 0.953 |
| Random Forest | 0.965 | 0.930 | 0.931 |

**Wine**

| Algorithme | Val Acc | Test Acc | Test F1 |
|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 1.000 | 0.963 | 0.962 |
| XGBoost | 0.963 | 1.000 | 1.000 |
| SGD | 1.000 | 0.926 | 0.926 |

### Champion retenu : Régression Logistique

Sur les deux datasets, la régression logistique obtient les meilleurs résultats en test, tout en étant le modèle le plus simple.

**Pourquoi ce choix va au-delà des chiffres :**

- **Explicabilité** : chaque coefficient du modèle correspond directement à une variable. On peut dire à un médecin ou à un client *pourquoi* la prédiction est telle qu'elle est. Une forêt aléatoire ou XGBoost sont des boîtes noires — performantes, mais muettes.
- **Type d'erreurs** : sur Breast Cancer, la régression logistique sort un score de probabilité (0 à 1). On peut ajuster le seuil de décision pour privilégier le rappel (ne rater aucun cancer) au détriment de la précision. Les arbres donnent des probabilités moins fiables.
- **Vitesse** : entraînement quasi-instantané, même sur de grands volumes. XGBoost et Gradient Boosting sont bien plus lents à tuner.
- **Robustesse** : avec une normalisation des données (déjà appliquée ici via `StandardScaler`), la régression logistique converge proprement et généralise bien sans hyperparamètres complexes à régler.

SGD est une alternative valide si le volume de données devient très grand (apprentissage incrémental possible), mais il est moins stable sur de petits datasets comme ceux-ci.
