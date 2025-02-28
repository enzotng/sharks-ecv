import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("Répertoire de travail actuel:", os.getcwd())

data = pd.read_csv("sharks_balanced_sample.csv")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc*100:.1f}%")

importances = model.feature_importances_
imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("Fichier 'feature_importances.png' sauvegardé.")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Country", order=data["Country"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Pays avec le plus d'attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_country.png")
print("Fichier 'attacks_by_country.png' sauvegardé.")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Area", order=data["Area"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Zones avec le plus d'attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_area.png")
print("Fichier 'attacks_by_area.png' sauvegardé.")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Year", order=sorted(data["Year"].dropna().unique()))
plt.xticks(rotation=45)
plt.title("Nombre d'attaques de requins par Année")
plt.tight_layout()
plt.savefig("attacks_by_year.png")
print("Fichier 'attacks_by_year.png' sauvegardé.")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="Activity", order=data["Activity"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Activités associées aux attaques de requins")
plt.tight_layout()
plt.savefig("attacks_by_activity.png")
print("Fichier 'attacks_by_activity.png' sauvegardé.")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="fatal", y="Age", data=data)
plt.title("Distribution de l'âge selon la fatalité")
plt.xlabel("Fatalité (0=Non, 1=Oui)")
plt.ylabel("Âge")
plt.tight_layout()
plt.savefig("age_distribution.png")
print("Fichier 'age_distribution.png' sauvegardé.")
plt.show()
