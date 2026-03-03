# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare machine learning models

# %%
import os
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from skrub.datasets import fetch_midwest_survey
from skore import EstimatorReport

# --- GESTION DES CHEMINS ---
# On définit le chemin vers le dossier racine du TP
BASE_DIR = Path(__file__).resolve().parent.parent

# %% [markdown]
# ## Load the dataset

# %%
dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# Simplification en classification binaire
y = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# Création des sets de test
sample_idx = X.sample(n=1000, random_state=1).index
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load the 3 models

# %%
# On utilise BASE_DIR pour localiser les fichiers .pkl à la racine du projet
try:
    model_lr = joblib.load(BASE_DIR / "model_logistic_regression.pkl")
    model_rf = joblib.load(BASE_DIR / "model_random_forest.pkl")
    model_gb = joblib.load(BASE_DIR / "model_gradient_boosting.pkl")
    print("Modèles chargés avec succès !")
except FileNotFoundError as e:
    print(f"Erreur : Impossible de trouver les fichiers .pkl dans {BASE_DIR}")
    raise e

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?

# %%
# Création des rapports avec skore
report_lr = EstimatorReport(model_lr, X_test=X_test, y_test=y_test)
report_rf = EstimatorReport(model_rf, X_test=X_test, y_test=y_test)
report_gb = EstimatorReport(model_gb, X_test=X_test, y_test=y_test)

def print_recall(name, report):
    # On extrait le recall pour la classe positive "North Central"
    metrics = report.metrics.summarize(pos_label="North Central").frame()
    recall_val = metrics.loc["recall"].iloc[0]
    print(f"Recall {name}: {recall_val:.3f}")

print_recall("Logistic Regression", report_lr)
print_recall("Random Forest", report_rf)
print_recall("Gradient Boosting", report_gb)

# %% [markdown]
# ## Question 7: Which model has the best practical application?
# 
# Gain = (TP * 5) + (TN * 2) - (FP * 10) - (FN * 1)

# %%
def calculate_gain(model, X_t, y_t):
    y_pred = model.predict(X_t)
    # On force l'ordre des labels pour que ravel() donne bien tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=["other", "North Central"]).ravel()
    return (tp * 5) + (tn * 2) - (fp * 10) - (fn * 1)

print(f"Gain LR: {calculate_gain(model_lr, X_test, y_test)}")
print(f"Gain RF: {calculate_gain(model_rf, X_test, y_test)}")
print(f"Gain GB: {calculate_gain(model_gb, X_test, y_test)}")

# %% [markdown]
# ## Question 8: Which model generalizes the best?

# %%
models = {"LR": model_lr, "RF": model_rf, "GB": model_gb}
summary = {}

for name, model in models.items():
    # cross_validate permet de calculer les scores train ET test
    cv = cross_validate(model, X, y, cv=5, return_train_score=True)
    summary[name] = {
        "Train Score (Mean)": cv['train_score'].mean(),
        "Test Score (Mean)": cv['test_score'].mean(),
        "Gap": cv['train_score'].mean() - cv['test_score'].mean()
    }

print(pd.DataFrame(summary).T)

# %% [markdown]
# ## Conclusion
#
# # My choice: Probablement le Gradient Boosting.
# # Reason: C'est souvent lui qui offre le meilleur score de test et le meilleur gain métier, 
# # même si la Régression Logistique a un "Gap" (overfitting) plus faible.