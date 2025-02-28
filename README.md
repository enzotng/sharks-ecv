# Projet de prédiction de la fatalité lors d'attaques de requins (ECV)

Ce projet a pour objectif de prédire si une attaque de requin est fatale ou non, à partir de diverses informations présentes dans un jeu de données. Différents scripts Python sont disponibles pour nettoyer les données, effectuer des validations croisées, créer des visualisations et expérimenter plusieurs approches de prédiction.

## Table des matières

1. [cleaningDataset.py](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#cleaningdatasetpy)
2. [crossValidationDataset.py](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#crossvalidationdatasetpy)
3. [illustrationDataset.py](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#illustrationdatasetpy)
4. [predictionDatasetV1.py](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#predictiondatasetv1py)
5. [predictionDatasetV2.py](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#predictiondatasetv2py)
6. [Fichiers de données](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#fichiers-de-donn%C3%A9es)
7. [Exécution du projet](https://chatgpt.com/c/67c1717e-f094-800e-addb-6714a5e38a26?model=o1#ex%C3%A9cution-du-projet)

---

## cleaningDataset.py

Ce script lit le fichier d’entrée `sharks.csv`, nettoie et normalise les données, puis crée un ensemble de données équilibré entre les attaques fatales et non fatales.

* Lecture des données brutes depuis `sharks.csv`.
* Suppression des colonnes en doublon et renommage des colonnes pour éviter les espaces.
* Remplissage des valeurs manquantes dans la colonne cible `Fatal (Y/N)` avec la modalité la plus fréquente.
* Pour les autres colonnes, les valeurs manquantes sont remplacées soit par la médiane (si la colonne est numérique) soit par la chaîne `"NULL"` (si elle est catégorielle).
* Conversion de la colonne `Fatal (Y/N)` en majuscules (`Y` ou `N`), puis filtrage des lignes n’ayant pas cette modalité.
* Échantillonnage équilibré (par défaut 500 observations max par classe) pour obtenir un dataset final avec deux classes de taille égale.
* Sauvegarde de l’ensemble de données nettoyé dans `cleanedDataset.csv`.

**Usage**

```bash
python cleaningDataset.py
```

Après exécution, le script affiche des informations sur la taille du dataset et sa répartition en classes, puis génère le fichier `cleanedDataset.csv`.

---

## crossValidationDataset.py

Ce script charge un ensemble de données (par défaut `cleanedDataset.csv` dans l’exemple) et compare plusieurs algorithmes de classification via une validation croisée.

* Lecture du dataset et prétraitement similaire (suppression de colonnes en double, valeurs manquantes, etc.).
* Conversion de la colonne cible en binaire (`fatal` = 1 si fatal, 0 sinon).
* Sélection d’un ensemble de features (par exemple `Age`, `Activity`, `Injury`, `Type`, `Country`, `Area`, `Species`, `Year`, `Sex`).
* Codage des variables catégorielles via `pd.get_dummies`.
* Mise en place de cinq modèles de classification : `KNeighborsClassifier`, `RandomForestClassifier`, `SVC`, `GradientBoostingClassifier`, `XGBClassifier`.
* Validation croisée (cv=5) pour chaque modèle et affichage de la précision moyenne.

**Usage**

```bash
python crossValidationDataset.py
```

Après exécution, le script affiche l’accuracy moyenne de chaque modèle.

---

## illustrationDataset.py

Ce script montre comment entraîner un modèle de classification (ici un `RandomForestClassifier`) et produire des visualisations descriptives à partir du dataset.

* Lecture du fichier `cleanedDataset.csv` (même logique de nettoyage et de transformation des colonnes).
* Séparation des features et de la cible, puis encodage des variables catégorielles.
* Entraînement d’un `RandomForestClassifier` et évaluation de son exactitude sur des données de test.
* Calcul et affichage de l’importance de chaque feature dans le modèle.
* Création de plusieurs graphiques descriptifs avec `seaborn` (par exemple la répartition par pays, par année, par activité, etc.).
* Export des graphiques au format PNG (fichiers comme `feature_importances.png`, `attacks_by_country.png`, etc.).

**Usage**

```bash
python illustrationDataset.py
```

Le script génère un ensemble d’images dans le répertoire de travail.

---

## predictionDatasetV1.py

Ce script explore différentes combinaisons de features ainsi que plusieurs hyperparamètres d’un `RandomForestClassifier` en parallèle. Il montre une approche plus exhaustive dans la recherche de la meilleure configuration.

* Lecture du dataset `cleanedDataset.csv`, nettoyage et transformation de la colonne cible.
* Création d’une liste de combinaisons de variables à partir de `["Age", "Activity", "Injury", "Type", "Country", "Area", "Species"]`.
* Sélection aléatoire de 20 combinaisons de features pour tester diverses configurations.
* Pour chaque combinaison, différentes valeurs de `n_estimators` et `max_depth` sont évaluées via un entraînement/test.
* Utilisation de la librairie `joblib` pour paralléliser les entraînements et accélérer la recherche de la meilleure accuracy.
* Affichage du temps estimé d’exécution et du temps réel.
* Identification de la meilleure combinaison de paramètres, puis affichage de l’accuracy optimale.
* Visualisation des résultats via un scatter plot (`hyperparameters_accuracy.png`).

**Usage**

```bash
python predictionDatasetV1.py
```

Le script lance plusieurs expérimentations et crée un graphe des accuracy en fonction des hyperparamètres.

---

## predictionDatasetV2.py

Ce script est une version simplifiée pour la recherche d’hyperparamètres d’un `RandomForestClassifier`.

* Lecture et nettoyage des données depuis `cleanedDataset.csv`.
* Séparation des features et de la cible (`fatal`).
* Test de différentes valeurs de `n_estimators` (1, 10, 500) et de `max_depth` (1, 32).
* Sélection de la meilleure configuration en fonction de l’accuracy.
* Affichage et sauvegarde de plusieurs graphiques descriptifs (répartition par pays, zone, année, etc.), identiques à ceux de la version précédente.

**Usage**

```bash
python predictionDatasetV2.py
```

Le script affiche directement l’accuracy pour différentes configurations, puis conserve la configuration gagnante.

---

## Fichiers de données

* **sharks.csv** : Fichier source contenant les données brutes.
* **cleanedDataset.csv** : Fichier nettoyé et équilibré généré par `cleaningDataset.py`.
* **cleanedDataset.csv** : Nom du dataset attendu par plusieurs scripts de prédiction (si vous utilisez `cleanedDataset.csv`, vous pouvez renommer le fichier ou adapter le code en conséquence).
* **list_coor_australia.csv** : Non utilisé explicitement dans les scripts ci-dessus, probablement un fichier contenant des coordonnées liées à certaines analyses cartographiques.

---

## Exécution du projet

1. Exécuter d’abord **cleaningDataset.py** pour générer un dataset propre et équilibré (`cleanedDataset.csv`).
2. Renommer le fichier de sortie en `cleanedDataset.csv` si nécessaire ou modifier les scripts suivants pour pointer vers le bon nom de fichier.
3. Tester différentes approches :
   * **crossValidationDataset.py** pour comparer plusieurs algorithmes via validation croisée.
   * **illustrationDataset.py** pour visualiser les résultats et observer l’importance des variables.
   * **predictionDatasetV1.py** et **predictionDatasetV2.py** pour expérimenter différents hyperparamètres d’un `RandomForestClassifier`.
4. Analyser les résultats et choisir la configuration la plus performante selon vos besoins.

---

*Note : Vérifiez que toutes les dépendances (pandas, sklearn, seaborn, matplotlib, joblib, xgboost, etc.) sont installées dans votre environnement Python (**Commande à éxécuter** : pip install pandas scikit-learn seaborn matplotlib joblib xgboost tqdm tqdm_joblib)*
