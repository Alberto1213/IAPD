from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_ml = pd.read_csv("../data/emag_phones_ml_ready.csv")

plt.figure()
sns.histplot(df_ml['price'], kde=True)
plt.title("Distribuția prețurilor telefoanelor")
plt.xlabel("Preț (lei)")
plt.ylabel("Numar de telefoane")
plt.show()

plt.figure()
sns.heatmap(df_ml[['price', 'storage', 'ram', "screen_size", "rating", "battery", "screen_megapixels"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Corelații principale intre variabile")
plt.show()

# Relația preț vs RAM
plt.figure()
sns.scatterplot(data=df_ml, x="ram", y="price")
plt.title("Preț în funcție de memoria RAM")
plt.show()

# Relația preț vs storage
plt.figure()
sns.scatterplot(data=df_ml, x="storage", y="price")
plt.title("Preț în funcție de Storage")
plt.show()

# Relația preț vs RAM
plt.figure()
sns.scatterplot(data=df_ml, x="screen_megapixels", y="price")
plt.title("Preț în funcție de rezolutie")
plt.show()

# --- 2. Analiză comparativă între mărci ---
brand_cols = [col for col in df_ml.columns if col.startswith("brand_")]

# Transformăm coloanele booleene într-o singură coloană "brand"
df_ml['brand'] = df_ml[brand_cols].idxmax(axis=1).str.replace('brand_', '')

# Preț mediu pe brand
brand_price_mean = df_ml.groupby('brand')['price'].mean().sort_values()
print(brand_price_mean)

# Vizualizare
plt.figure()
sns.barplot(x=brand_price_mean.index, y=brand_price_mean.values)
plt.title("Prețul mediu per brand")
plt.xlabel("Brand")
plt.ylabel("Preț mediu (€)")
plt.xticks(rotation=45)
plt.show()

# Variabile numerice
X_cluster = df_ml[['price', 'storage', 'ram', 'screen_size', 'rating', 'battery', 'screen_megapixels']]

# Normalizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means cu 3 clustere
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_ml['cluster'] = clusters

# Vizualizare 2D (cu PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_ml['cluster'], palette='Set2')
plt.title("Clusterele telefoanelor (K-Means)")
plt.show()

df_ml = df_ml.drop_duplicates()
print("Număr de rânduri după eliminarea duplicatelor:", len(df_ml))

print(df_ml.groupby('brand')['price'].nunique())

features = [
    "storage", "ram", "battery",
    "screen_megapixels", "screen_size", "rating"
]

for col in features:
    df_ml[col + "_norm"] = (df_ml[col] - df_ml[col].min()) / (df_ml[col].max() - df_ml[col].min())

df_ml["scor_calitate"] = (
        0.25 * df_ml["ram_norm"] +
        0.25 * df_ml["storage_norm"] +
        0.1 * df_ml["battery_norm"] +
        0.30 * df_ml["rating_norm"] +
        0.05 * df_ml["screen_size_norm"] +
        0.05 * df_ml["screen_megapixels_norm"]
)

df_ml["raport_calitate_pret"] = (df_ml["scor_calitate"] / df_ml["price"]) * 1000

top_perf = df_ml.sort_values(
    "raport_calitate_pret", ascending=False).head(30)

print(top_perf[
          ["brand", "price", "scor_calitate", "raport_calitate_pret"]
      ])

plt.figure()
sns.barplot(
    x="brand",
    y="raport_calitate_pret",
    data=top_perf
)
plt.title("Top 10 telefoane – raport calitate/preț")
plt.xticks(rotation=45)
plt.show()

# On s'assure que le DataFrame est trié
top_perf_sorted = top_perf.sort_values(by='raport_calitate_pret', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x='raport_calitate_pret',
    y='brand',
    data=top_perf_sorted,
    palette='viridis',
    hue='brand'
)

# Ajout d'étiquettes (prix) à côté de chaque barre
for index, value in enumerate(top_perf_sorted['raport_calitate_pret']):
    price = top_perf_sorted['price'].iloc[index]
    brand = top_perf_sorted['brand'].iloc[index]
    ax.text(
        value + 0.02,  # position horizontale (légèrement à droite de la barre)
        index,  # position verticale alignée sur la barre
        f"{price:.0f} €",  # texte affiché
        va='center',
        ha='left',
        fontsize=9
    )

plt.title("Top 10 telefoane cu cel mai bun raport calitate/preț", fontsize=14)
plt.xlabel("Raport calitate/preț")
plt.ylabel("Brand")
plt.tight_layout()
plt.show()

# Facem o copie a bazei de date
df_model = df_ml.copy()

# 1️⃣ Transformăm coloanele text (ex: brand) în valori numerice
categorical_cols = df_model.select_dtypes(include=['object']).columns
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# 2️⃣ Definim variabilele explicative (X) și ținta (y)
X = df_model.drop(columns=['price'])
y = df_model['price']

# 3️⃣ Împărțim în seturi de antrenare și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Antrenăm modelul
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluăm performanța
y_pred = model.predict(X_test)
print("Eroare medie absolută (MAE):", mean_absolute_error(y_test, y_pred))
print("Coeficient de determinare (R²):", r2_score(y_test, y_pred))

print("R² train:", model.score(X_train, y_train))
print("R² test:", model.score(X_test, y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# 6️⃣ Prezicem prețul pentru FIECARE rând (telefon) din dataset
df_model['pret_prezis'] = model.predict(X)

# 7️⃣ Adăugăm rezultatul în baza originală (df_ml)
df_ml['pret_prezis'] = df_model['pret_prezis']

# --- Rezultat: primele 10 telefoane cu preț real vs preț prezis ---
rezultat = df_ml[['brand', 'ram', 'storage', 'price', 'pret_prezis']].head(20)
print(rezultat)

X_no_brand = X.drop(columns=[c for c in X.columns if c.startswith("brand_")])

X_train, X_test, y_train, y_test = train_test_split(
    X_no_brand, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("R² fără brand:", model.score(X_test, y_test))

df_ml['diferenta'] = df_ml['pret_prezis'] - df_ml['price']

# Cele mai subevaluate (preț real < preț prezis)
subevaluate = df_ml.sort_values('diferenta').head(10)

# Cele mai supraevaluate (preț real > preț prezis)
supraevaluate = df_ml.sort_values('diferenta', ascending=False).head(10)

print("🔹 Telefoane subevaluate:")
print(subevaluate[['brand', 'price', 'pret_prezis', 'diferenta']])

print("🔹 Telefoane supraevaluate:")
print(supraevaluate[['brand', 'price', 'pret_prezis', 'diferenta']])

plt.figure(figsize=(8, 6))
plt.scatter(df_ml['price'], df_ml['pret_prezis'], alpha=0.7)
plt.plot(
    [df_ml['price'].min(), df_ml['price'].max()],
    [df_ml['price'].min(), df_ml['price'].max()],
    color='red', linestyle='--'
)
plt.title("Preț real vs. Preț prezis (Random Forest)", fontsize=14)
plt.xlabel("Preț real (lei)")
plt.ylabel("Preț prezis (lei)")
plt.tight_layout()
plt.show()