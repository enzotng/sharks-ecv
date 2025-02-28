import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("cleanedDataset.csv")
data.columns = data.columns.str.strip()
data = data.loc[:, ~data.columns.duplicated()]
data = data.loc[:, data.columns != '']
data["Fatal (Y/N)"] = data["Fatal (Y/N)"].fillna(data["Fatal (Y/N)"].mode()[0]).astype(str).str.strip().str.upper()
data = data[data["Fatal (Y/N)"].isin(["Y", "N"])]
data.rename(columns={"Fatal (Y/N)": "fatal"}, inplace=True)
data["fatal"] = data["fatal"].map({"Y": 1, "N": 0})

features = ["Age", "Activity", "Injury", "Type", "Country", "Area", "Species", "Year", "Sex"]
y = data["fatal"]
x = data[features]
X = pd.get_dummies(x, drop_first=True)
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col].fillna(X[col].mean(), inplace=True)
    else:
        X[col].fillna("NULL", inplace=True)

results_list = []
for n_estimators in [1, 10, 500]:
    for max_depth in [1, 32]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results_list.append({"n_estimators": n_estimators, "max_depth": max_depth, "accuracy": acc})
        print(f"n_estimators {n_estimators} et max_depth {max_depth} : Accuracy = {acc*100:.1f}%")

df_results = pd.DataFrame(results_list)
best_row = df_results.loc[df_results["accuracy"].idxmax()]
best_params = {"n_estimators": int(best_row["n_estimators"]), "max_depth": int(best_row["max_depth"])}
print("Meilleurs hyperparamètres :", best_params)
print(f"Meilleure accuracy : {best_row['accuracy']*100:.1f}%")

model = RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy du modèle final: {acc*100:.1f}%")

importances = model.feature_importances_
imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Country", order=data["Country"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Pays avec le plus d'attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_country.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Area", order=data["Area"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Zones avec le plus d'attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_area.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Year", order=sorted(data["Year"].dropna().unique()))
plt.xticks(rotation=45)
plt.title("Nombre d'attaques de requins par Année")
plt.tight_layout()
plt.savefig("attacks_by_year.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Activity", order=data["Activity"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Activités associées aux attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_activity.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="fatal", y="Age", data=data)
plt.title("Distribution de l'âge selon la fatalité")
plt.xlabel("Fatalité (0=Non, 1=Oui)")
plt.ylabel("Âge")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()
