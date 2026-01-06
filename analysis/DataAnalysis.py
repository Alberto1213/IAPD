from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Citire date ---
df_ml = pd.read_csv("../data/emag_phones_ml_ready.csv")

# --- Vizualizări inițiale ---
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

plt.figure()
sns.scatterplot(data=df_ml, x="ram", y="price")
plt.title("Preț în funcție de memoria RAM")
plt.show()

plt.figure()
sns.scatterplot(data=df_ml, x="storage", y="price")
plt.title("Preț în funcție de Storage")
plt.show()

plt.figure()
sns.scatterplot(data=df_ml, x="screen_megapixels", y="price")
plt.title("Preț în funcție de rezolutie")
plt.show()

# --- Analiză comparativă între branduri ---
brand_cols = [col for col in df_ml.columns if col.startswith("brand_")]
df_ml['brand'] = df_ml[brand_cols].idxmax(axis=1).str.replace('brand_', '')
df_ml['brand_original'] = df_ml['brand']  # păstrăm pentru vizualizare și top

brand_price_mean = df_ml.groupby('brand')['price'].mean().sort_values()
plt.figure()
sns.barplot(x=brand_price_mean.index, y=brand_price_mean.values)
plt.title("Prețul mediu per brand")
plt.xlabel("Brand")
plt.ylabel("Preț mediu (€)")
plt.xticks(rotation=45)
plt.show()

# --- Clustering KMeans ---
features_cluster = ['price', 'storage', 'ram', 'screen_size', 'rating', 'battery', 'screen_megapixels']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml[features_cluster])

kmeans = KMeans(n_clusters=3, random_state=42)
df_ml['cluster'] = kmeans.fit_predict(X_scaled)

# PCA pentru vizualizare
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_ml['cluster'], palette='Set2')
plt.title("Clusterele telefoanelor (K-Means)")
plt.show()

# --- Scor calitate și raport calitate/preț ---
for col in ["storage", "ram", "battery", "screen_megapixels", "screen_size", "rating"]:
    df_ml[col + "_norm"] = (df_ml[col] - df_ml[col].min()) / (df_ml[col].max() - df_ml[col].min())

df_ml["scor_calitate"] = (
    0.20 * df_ml["ram_norm"] +
    0.25 * df_ml["storage_norm"] +
    0.15 * df_ml["battery_norm"] +
    0.15 * df_ml["rating_norm"] +
    0.10 * df_ml["screen_size_norm"] +
    0.15 * df_ml["screen_megapixels_norm"]
)

df_ml["raport_calitate_pret"] = (df_ml["scor_calitate"] / df_ml["price"]) * 1000
top_perf = df_ml.sort_values("raport_calitate_pret", ascending=False).head(30)

plt.figure()
sns.barplot(x="brand", y="raport_calitate_pret", data=top_perf)
plt.title("Top telefoane – raport calitate/preț")
plt.xticks(rotation=45)
plt.show()

# --- Pregătire date pentru ML ---
df_model = df_ml.copy()

# Coloane textuale pentru get_dummies (excluzând brand_original)
categorical_cols = df_model.select_dtypes(include=['object']).columns
categorical_cols = [c for c in categorical_cols if c != 'brand_original']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Variabile explicative și target
X = df_model.drop(columns=['price', 'brand_original', 'scor_calitate', 'raport_calitate_pret', "cluster", "storage",
                           "ram", "battery", "screen_megapixels", "screen_size", "rating"])
y = df_model['price']
# Împărțire train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modele ML ---
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "Linear Regression": LinearRegression(),
    "HistGradient Boosting": HistGradientBoostingRegressor(max_iter=200, random_state=42),
}

results = []

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    df_model[name + "_pred"] = model.predict(X)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² train:", model.score(X_train, y_train))
    print("R² test:", model.score(X_test, y_test))

    # Diferența între preț prezis și real
    df_model['diferenta'] = df_model[name + "_pred"] - df_model['price']

    # Telefoane subevaluate/supraevaluate
    subevaluate = df_model.sort_values('diferenta').head(10)
    supraevaluate = df_model.sort_values('diferenta', ascending=False).head(10)

    print("🔹 Telefoane subevaluate:")
    print(subevaluate[['brand_original', 'price', name + "_pred", 'diferenta']])
    print("🔹 Telefoane supraevaluate:")
    print(supraevaluate[['brand_original', 'price', name + "_pred", 'diferenta']])

    # Plot preț real vs prezis
    plt.figure(figsize=(8, 6))
    plt.scatter(df_model['price'], df_model[name + "_pred"], alpha=0.7)
    plt.plot([df_model['price'].min(), df_model['price'].max()],
             [df_model['price'].min(), df_model['price'].max()],
             color='red', linestyle='--')
    plt.title(f"Preț real vs prezis ({name})")
    plt.xlabel("Preț real (lei)")
    plt.ylabel("Preț prezis (lei)")
    plt.tight_layout()
    plt.show()

    if name != "Linear Regression" and name != "HistGradient Boosting":
        importances = model.feature_importances_
        features = X.columns
        feat_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
        feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feat_importance_df.head(3))
        plt.title("Top 3 factori care influențează prețul telefoanelor")
        plt.show()

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print("=" * 60)
    print("🔄 CROSS-VALIDATION (5-fold):")
    print("=" * 60)
    print(f"R² per fold: {cv_scores}")
    print(f"R² mediu: {cv_scores.mean():.4f}")
    print(f"Std dev: {cv_scores.std():.4f}")
    print("=" * 60)

    # Interpretare
    if cv_scores.mean() > 0.90 and cv_scores.std() < 0.05:
        print("✅ MODEL EXCELENT ȘI CONSISTENT!")
    elif cv_scores.mean() > 0.85:
        print("✅ MODEL FOARTE BUN!")
    elif cv_scores.std() > 0.15:
        print("⚠️ Variație mare între folduri - instabil")

# --- Rezumat metrici ---
results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
print("\n=== Comparatie modele ===")
print(results_df)
