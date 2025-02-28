Le dataset, qu’il s’agisse de l’attaque de requins ou du Titanic, est couramment divisé en deux parties : **80 %** des données pour l’entraînement et **20 %** pour le test. Cette répartition permet de construire un modèle et d’évaluer sa performance sur des données encore jamais vues par l’algorithme, afin de s’assurer qu’il généralise correctement et ne se contente pas de mémoriser l’ensemble d’entraînement.

### À propos du dataset

* Dans le cas du  **Titanic** , on retrouve des informations comme l’âge des passagers (`Age`), leur sexe (`Sex`), leur classe de voyage (`Pclass`), le port d’embarquement (`Embarked`), etc. La variable cible (`Survived`) indique si une personne a survécu (1) ou non (0).
* Pour l’attaque de requins, les colonnes peuvent inclure l’activité de la victime (`Activity`), le pays (`Country`), la zone côtière (`Area`), l’espèce du requin (`Species`), l’âge de la victime (`Age`) ou encore la gravité de l’attaque (`Fatal (Y/N)`).

Dans les deux contextes, on collecte et nettoie les données (traitement des valeurs manquantes, encodage des variables catégorielles) afin de préparer un jeu de données utilisable par les algorithmes de machine learning.

### Modèles couramment utilisés

1. **RandomForestClassifier**
   * Le random forest est un ensemble d’arbres de décision. Chaque arbre est construit sur un échantillon aléatoire (bootstrap) des données et un sous-ensemble des variables.
   * Le vote majoritaire (ou la moyenne des probabilités) de tous les arbres détermine la prédiction finale.
   * Cette approche est robuste contre le surapprentissage (overfitting) et fonctionne aussi bien sur des données tabulaires avec de nombreuses variables numériques et/ou catégorielles.
2. **XGBoost**
   * XGBoost (Extreme Gradient Boosting) est également basé sur des arbres de décision, mais suit une logique de boosting.
   * Les arbres sont construits de manière séquentielle, chaque nouvel arbre cherchant à corriger les erreurs du précédent.
   * XGBoost est réputé pour être très performant, notamment sur de nombreux jeux de données tabulaires (y compris Titanic), mais il exige souvent un peu plus de travail de tuning (réglage d’hyperparamètres).
3. **KNN (K-Nearest Neighbors)**
   * KNN est un algorithme plus simple : pour chaque point à prédire, il recherche les k voisins les plus proches dans l’espace des features et effectue une « majorité » sur leurs étiquettes.
   * Ses performances dépendent beaucoup des échelles de données (il est sensible aux grandeurs numériques très différentes), du choix de k, ainsi que de la distance utilisée (Euclidienne, Manhattan, etc.).

### Exemple avec le Titanic

Dans le dataset du Titanic (issu de Kaggle), la variable cible est `Survived`. On a notamment constaté que :

* **Age** : Les personnes plus jeunes avaient plus de chances de survie, possiblement à cause des priorités d’évacuation ou de leur capacité à se déplacer plus rapidement.
* **Sex** : Les femmes avaient un taux de survie supérieur, correspondant à la politique « les femmes et les enfants d’abord ».

D’autres variables (comme la **classe** du passager ou l’information du  **port d’embarquement** ) ont également un impact, mais dans les analyses effectuées, **l’âge** et **le sexe** se sont révélés décisifs parmi les critères disponibles.

### Conclusion

Quelle que soit la nature du dataset, la démarche est similaire :

1. Nettoyer et préparer les données (répartition 80 % entraînement / 20 % test).
2. Tester différents algorithmes (random forest, xgboost, knn, etc.).
3. Régler les hyperparamètres (par exemple, le nombre d’arbres ou la profondeur maximale d’un arbre).
4. Comparer leurs performances sur l’ensemble de test pour identifier la solution la plus adaptée.
