# Prédiction de revenu — Notebook (rendu01.ipynb) 

Ce notebook construit un modèle de prédiction du revenu à partir du dataset “Adult”, puis évalue à la fois la performance et des métriques de fairness (avec un focus sur le genre). 

## Vue d’ensemble du notebook 
- Chargement du dataset depuis `adult.csv` dans un DataFrame pandas. 
- Inspection rapide des données (ex. `data.head()` avec des colonnes comme `age`, `workclass`, `education`, `gender`, `native-country`, `income`). 
- Gestion des valeurs manquantes en remplaçant `"?"` par `NaN`. 

## Nettoyage & feature engineering 
- Encodage des variables catégorielles via one-hot encoding (`pd.get_dummies`) pour `workclass`, `marital-status`, `occupation`, `relationship`. 
- Suppression de `education` et conservation de `educational-num` (éducation déjà encodée numériquement).
- Binarisation de `race` (`White` -> 1, autres -> 0) et de `gender` (`Female` -> 0, `Male` -> 1). 
- Regroupement de `native-country` en `United-States` vs `ExPat`, puis conversion binaire.
- Binarisation de la cible `income` (`<=50K` -> 0, `>50K` -> 1). 

## Métriques de fairness implémentées
- Définition d’un score de **Disparate Impact** basé sur les taux de prédictions positives par genre. 
- Définition d’un score de **Zemel fairness** basé sur l’écart de taux de prédictions positives entre genres. 
- Calcul du **Balanced Error Rate (BER)** à partir de la matrice de confusion. 

## Entraînement & évaluation (baseline) 
- Normalisation des features avec `MinMaxScaler`. 
- Split train/test avec `train_test_split(test_size=0.3, random_state=42, stratify=gender)`. 
- Entraînement d’un `RandomForestClassifier(class_weight="balanced_subsample", random_state=42)` et calcul de métriques (accuracy, ROC-AUC, sensibilité, spécificité, BER).
- Affichage des scores Disparate Impact et Zemel Fairness sur les prédictions du test set. 

## Expériences de réduction de biais [file:83]
- Sur-échantillonnage (upsampling) du genre minoritaire via `sklearn.utils.resample`, puis ré-entraînement et ré-évaluation. 
- Application d’une transformation de dataset de type “Kamiran and Calders” (stockée dans `D`) pour réduire la discrimination avant l’entraînement.

## Fichiers produits par le notebook 
- Export d’un dataset nettoyé vers `adultclean.csv`. 
- Export d’un dataset “fair” vers `trainfair.csv`. 
