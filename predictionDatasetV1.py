import pandas as pd
import itertools
import time
import random
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('sharks_balanced_sample.csv')
data.columns = data.columns.str.strip()
data = data.loc[:, ~data.columns.duplicated()]
data = data.loc[:, data.columns != '']
data["Fatal (Y/N)"] = data["Fatal (Y/N)"].fillna(data["Fatal (Y/N)"].mode()[0]).astype(str).str.strip().str.upper()
data = data[data["Fatal (Y/N)"].isin(["Y", "N"])]
data.rename(columns={"Fatal (Y/N)": "fatal"}, inplace=True)
data["fatal"] = data["fatal"].map({"Y": 1, "N": 0})
features = ["Age", "Activity", "Injury", "Type", "Country", "Area", "Species"]
all_combos = [combo for r in range(1, len(features) + 1) for combo in itertools.combinations(features, r)]
random.seed(42)
selected_combos = random.sample(all_combos, 20)
X_dict = {combo: pd.get_dummies(data[list(combo)], drop_first=True) for combo in selected_combos}
n_estimators_list = [10, 50, 100, 200, 500]
max_depth_list = [5, 10, 20, 30, 50]
random_states = [42]
total_iterations = len(X_dict) * len(n_estimators_list) * len(max_depth_list) * len(random_states)
print("Nombre total d'itérations :", total_iterations)
def evaluate(combo, n_estimators, max_depth, random_state):
    X = X_dict[combo]
    y = data["fatal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc, combo, n_estimators, max_depth, random_state
if __name__ == '__main__':
    sample_iterations = 10
    sample_params = [(combo, n_estimators_list[0], max_depth_list[0], 42) for combo in list(X_dict.keys())[:sample_iterations]]
    start_sample = time.time()
    [evaluate(*params) for params in sample_params]
    sample_time = time.time() - start_sample
    avg_time = sample_time / sample_iterations
    estimated_total_time = total_iterations * avg_time
    print("Temps estimé pour l'exécution complète : {:.2f} secondes".format(estimated_total_time))
    start_full = time.time()
    with tqdm_joblib(tqdm(total=total_iterations, desc="Évaluation")):
        results = Parallel(n_jobs=-1)(
            delayed(evaluate)(combo, n_estimators, max_depth, 42)
            for combo in X_dict.keys()
            for n_estimators in n_estimators_list
            for max_depth in max_depth_list
            for random_state in [42]
        )
    elapsed_time = time.time() - start_full
    best_result = max(results, key=lambda x: x[0])
    best_accuracy, best_combo, best_n_estimators, best_max_depth, best_random_state = best_result
    print("Meilleure combinaison de features :", best_combo)
    print("Meilleurs hyperparamètres : n_estimators =", best_n_estimators, ", max_depth =", best_max_depth, ", random_state =", best_random_state)
    print("Meilleure accuracy obtenue : {:.1f}%".format(best_accuracy * 100))
    print("Temps d'exécution réel : {:.2f} secondes".format(elapsed_time))
    results_df = pd.DataFrame(results, columns=["accuracy", "combo", "n_estimators", "max_depth", "random_state"])
    results_df["max_depth_str"] = results_df["max_depth"].apply(lambda x: "None" if x is None else x)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="n_estimators", y="max_depth_str", hue="accuracy", size="accuracy", palette="viridis", sizes=(20, 200))
    plt.title("Accuracy vs Hyperparameters")
    plt.tight_layout()
    plt.savefig("hyperparameters_accuracy.png")
    plt.show()
