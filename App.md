
# Prédiction de revenu — Application

Cette application permet de prédire une classe de revenu à partir d’informations saisies par l’utilisateur, en réutilisant le modèle entraîné dans le notebook et les mêmes règles de préparation des données.

## Principe de fonctionnement
1. **Saisie des informations**
L’utilisateur renseigne les variables décrivant un individu (ex. âge, niveau d’éducation, profession, pays, etc.), via une interface simple (souvent une interface web).  

2. **Prétraitement identique au notebook**
Avant de prédire, l’application transforme les entrées exactement comme dans le notebook afin que les “features” aient le même format que pendant l’entraînement (ex. encodage des catégories en one-hot, binarisation de certaines variables comme le genre/pays, conversion de la cible en 0/1 côté entraînement).

3. **Inférence (prédiction)**
Le modèle reçoit le vecteur de features prétraité et renvoie une prédiction (classe de revenu) et éventuellement un score associé (probabilité ou confiance selon l’implémentation).  

4. **Affichage du résultat**
L’application affiche le résultat de manière lisible (par exemple “<= 50K” vs “> 50K”, ou “0/1” )
