import pandas as pd

# Lecture et préparation du dataset
df = pd.read_csv("sharks.csv")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()]
df = df.loc[:, df.columns != '']

# Remplissage des valeurs manquantes pour toutes les colonnes
for col in df.columns:
    if col == "Fatal (Y/N)":
        # Pour la cible, on remplit avec la modalité la plus fréquente
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("NULL")
        else:
            df[col] = df[col].fillna(df[col].median())

# Prétraitement de la colonne "Fatal (Y/N)"
df["Fatal (Y/N)"] = df["Fatal (Y/N)"].astype(str).str.strip().str.upper()
df = df[df["Fatal (Y/N)"].isin(["Y", "N"])]

# Séparation en deux groupes selon la classe "Fatal (Y/N)"
groupY = df[df["Fatal (Y/N)"] == "Y"]
groupN = df[df["Fatal (Y/N)"] == "N"]

# Limiter le nombre d'exemples par classe (max 3000 par classe pour un total < 6000 lignes)
max_per_class = 3000
n_sample = min(len(groupY), len(groupN), max_per_class)

# Échantillonnage aléatoire équilibré
sampleY = groupY.sample(n=n_sample, random_state=42)
sampleN = groupN.sample(n=n_sample, random_state=42)

balanced_df = pd.concat([sampleY, sampleN]).sample(frac=1, random_state=42).reset_index(drop=True)

# Affichage des informations et sauvegarde du CSV
print("Nombre total de lignes :", len(balanced_df))
print("Répartition des classes :")
print(balanced_df["Fatal (Y/N)"].value_counts())

balanced_df.to_csv("sharks_balanced_sample.csv", index=False)
print("CSV sauvegardé sous le nom 'sharks_balanced_sample.csv'")
