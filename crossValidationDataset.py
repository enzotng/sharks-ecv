import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Accuracy moyenne pour le modèle {name}: {scores.mean() * 100:.1f}%")
    print("---------------------------------------------------------------------")
