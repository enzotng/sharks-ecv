import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("sharks.csv")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]
df = df.loc[:, df.columns != '']

df["Fatal (Y/N)"] = df["Fatal (Y/N)"].fillna(df["Fatal (Y/N)"].mode()[0]).astype(str).str.strip().str.upper()
df = df[df["Fatal (Y/N)"].isin(["Y", "N"])]
df.rename(columns={"Fatal (Y/N)": "fatal"}, inplace=True)
df["fatal"] = df["fatal"].map({"Y": 1, "N": 0})

features = df.columns.drop("fatal")
X = df[features]
Y = df["fatal"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

imputer_num = SimpleImputer(strategy="median")
imputer_cat = SimpleImputer(strategy="most_frequent")
X[num_cols] = imputer_num.fit_transform(X[num_cols])
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

n_estimators_list = [50, 100, 200]
max_depth_list = [None, 5, 10]
random_state_list = [0, 42]

best_accuracy = 0
best_config = None
best_k = None

for k in range(1, X_train.shape[1] + 1):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    for n in n_estimators_list:
        for d in max_depth_list:
            for r in random_state_list:
                clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=r)
                clf.fit(X_train_sel, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test_sel))
                print(f"k: {k}, n_estimators: {n}, max_depth: {d}, random_state: {r} -> Accuracy: {acc * 100:.1f}%")
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_config = (n, d, r)
                    best_k = k

selector = SelectKBest(score_func=f_classif, k=best_k)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]

print("\nMeilleure configuration:")
print(f"Nombre de features sélectionnées (k): {best_k}")
print("Features sélectionnées :", list(selected_features))
print(f"n_estimators: {best_config[0]}, max_depth: {best_config[1]}, random_state: {best_config[2]} -> Accuracy: {best_accuracy * 100:.1f}%")
