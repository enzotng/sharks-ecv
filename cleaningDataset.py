import pandas as pd

df = pd.read_csv("sharks.csv")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]
df = df.loc[:, df.columns != '']

for col in df.columns:
    if col == "Fatal (Y/N)":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("NULL")
        else:
            df[col] = df[col].fillna(df[col].median())

df["Fatal (Y/N)"] = df["Fatal (Y/N)"].astype(str).str.strip().str.upper()
df = df[df["Fatal (Y/N)"].isin(["Y", "N"])]

groupY = df[df["Fatal (Y/N)"] == "Y"]
groupN = df[df["Fatal (Y/N)"] == "N"]

max_per_class = 500
n_sample = min(len(groupY), len(groupN), max_per_class)

sampleY = groupY.sample(n=n_sample, random_state=42)
sampleN = groupN.sample(n=n_sample, random_state=42)

balanced_df = pd.concat([sampleY, sampleN]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Nombre total de lignes :", len(balanced_df))
print("Répartition des classes :")
print(balanced_df["Fatal (Y/N)"].value_counts())

balanced_df.to_csv("cleanedDataset.csv", index=False)
print("CSV sauvegardé sous le nom 'cleanedDataset.csv'")
