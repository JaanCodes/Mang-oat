# --- [PASO 0: Importaciones] ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
import warnings

# Modelos
import lightgbm as lgb
from catboost import CatBoostRegressor

# Preprocessing y Pipelines
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# Configuración
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
warnings.filterwarnings('ignore')

print("Todas las librerías importadas.")

# --- [PASO 1: Carga de Datos] ---
print("--- Cargando Datos ---")
data_dir = Path('.')
train_path = data_dir / 'train.csv'
test_path = data_dir / 'test.csv'

try:
    train_df = pd.read_csv(train_path, sep=';')
    test_df = pd.read_csv(test_path, sep=';')
except FileNotFoundError:
    print("Error: Ficheros 'train.csv' o 'test.csv' no encontrados.")
    exit()

print(f"Datos cargados: Train={train_df.shape}, Test={test_df.shape}")

# --- [PASO 2: Agregación de Datos] ---
print("--- Agregando Datos (groupby ID) ---")
# Target
y_train_agg = train_df.groupby('ID')['weekly_demand'].sum()
# Features
X_train_agg_full = train_df.groupby('ID').first()

# Alinear columnas entre train y test
train_features = set(X_train_agg_full.columns)
test_features = set(test_df.columns)
common_columns = list(train_features.intersection(test_features))

X = X_train_agg_full[common_columns].copy()
test_ids_for_submission = test_df['ID']
X_test = test_df[common_columns].copy()

# Alinear target con features
y = y_train_agg.reindex(X.index)

print(f"Formas Agregadas: X={X.shape}, y={y.shape}, X_test={X_test.shape}")


# --- [PASO 3: Ingeniería de Fechas y Estacionalidad (Fourier)] ---
print("--- Creando features de Fourier (sin/cos) para 'start_month' ---")

def create_temporal_features(df):
    """Crea features de mes y las transforma en sin/cos."""
    df_copy = df.copy()
    
    # 1. Crear 'start_month' (1-12)
    df_copy['phase_in'] = pd.to_datetime(df_copy['phase_in'], errors='coerce')
    df_copy['start_month'] = df_copy['phase_in'].dt.month
    
    # 2. Rellenar nulos del mes
    # Usamos 6.5 (entre junio y julio) como mediana neutral
    median_month = 6.5 
    df_copy['start_month'] = df_copy['start_month'].fillna(median_month)

    # 3. Crear Features de Fourier (La "Sinuide")
    # Esto convierte el mes (1-12) en un círculo perfecto
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['start_month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['start_month'] / 12)
    
    # 4. Eliminar la columna original, ya no es necesaria
    df_copy = df_copy.drop('start_month', axis=1)
    
    return df_copy

X = create_temporal_features(X)
X_test = create_temporal_features(X_test)
print("Features 'month_sin' y 'month_cos' creadas.")


# --- [PASO 4: Ingeniería de Embeddings Estacionales (Tu función)] ---
print("--- Creando features de embeddings estacionales ---")

def create_embedding_features(df, embedding_col='image_embedding'):
    """Crea features de clustering y similitud por temporada."""
    print(f"Procesando embeddings para crear features estacionales... (Forma: {df.shape})")
    df = df.copy()
    
    # Rellenar NaNs en 'id_season' antes de usarlo para agrupar
    if df['id_season'].isnull().any():
        print("Rellenando NaNs en 'id_season' con -1 (temporada desconocida)")
        df['id_season'] = df['id_season'].fillna(-1).astype(int)

    # Parsear los strings de embeddings a arrays de numpy
    df['embedding_parsed'] = df[embedding_col].fillna('').apply(
        lambda x: np.array([float(v) for v in x.split(',') if v.strip()]) if x else np.zeros(256)
    )

    # Asegurar que todos los embeddings tengan la misma longitud (padding)
    max_len = max(len(e) for e in df['embedding_parsed'])
    max_len = max(max_len, 256) # Asegurar al menos 256
    
    df['embedding_parsed'] = df['embedding_parsed'].apply(
        lambda x: np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else x[:max_len]
    )

    embeddings_matrix = np.stack(df['embedding_parsed'].values)
    
    # Inicializar nuevas columnas
    df['embedding_cluster'] = -1
    df['similarity_to_season_center'] = 0.0
    df['avg_distance_within_season'] = 0.0
    df['similarity_to_nearest_seasons'] = 0.0

    unique_seasons = df['id_season'].unique()
    
    # --- Clustering por temporada ---
    season_clusters = {}
    for season in unique_seasons:
        if season == -1: continue # Ignorar 'temporada desconocida'
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 5: # Solo si hay suficientes muestras
            n_clusters = min(5, len(season_embeddings) // 10 + 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(season_embeddings)
            df.loc[season_mask, 'embedding_cluster'] = clusters
            season_clusters[season] = kmeans.cluster_centers_

    # --- Centroides y similitud ---
    season_centroids = {}
    for season in unique_seasons:
        if season == -1: continue
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 0:
            season_centroids[season] = season_embeddings.mean(axis=0)

    for season, centroid in season_centroids.items():
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 0:
            sims = cosine_similarity(season_embeddings, centroid.reshape(1, -1)).flatten()
            df.loc[season_mask, 'similarity_to_season_center'] = sims

    # --- Distancias intra-temporada ---
    for season in unique_seasons:
        if season == -1: continue
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 1: # Necesitas al menos 2 para comparar
            distances = cdist(season_embeddings, season_embeddings, metric='euclidean')
            # Distancia media (dividido por N-1)
            avg_dists = distances.sum(axis=1) / (len(season_embeddings) - 1)
            df.loc[season_mask, 'avg_distance_within_season'] = avg_dists

    # --- Similitud entre temporadas (tendencias) ---
    if len(season_centroids) > 1:
        centroid_matrix = np.array(list(season_centroids.values()))
        
        for idx, (season, centroid) in enumerate(season_centroids.items()):
            season_mask = (df['id_season'] == season)
            # Comparar con todos los *otros* centroides
            other_centroids = np.delete(centroid_matrix, idx, axis=0)
            if other_centroids.shape[0] > 0:
                sims = cosine_similarity(centroid.reshape(1, -1), other_centroids).flatten()
                # Similitud media con las 3 temporadas más parecidas
                top_sims = np.sort(sims)[-3:] if len(sims) >= 3 else sims
                df.loc[season_mask, 'similarity_to_nearest_seasons'] = top_sims.mean()

    df = df.drop('embedding_parsed', axis=1)
    print("Features de embeddings estacionales creadas.")
    return df

X = create_embedding_features(X)
X_test = create_embedding_features(X_test)


# --- [PASO 5: Definición de Features y Pipelines] ---
print("--- Definiendo listas de Features y Pipelines ---")

# Función para parsear embeddings (para PCA)
def parse_embeddings(df_column):
    embeddings_series = df_column.iloc[:, 0].fillna('')
    embeddings_list = embeddings_series.apply(
        lambda x: [float(v) for v in x.split(',') if v.strip()] if x else []
    )
    try:
        # Encontrar la primera dimensión válida
        target_dim = len(next(item for item in embeddings_list if item))
    except StopIteration:
        target_dim = 256 # Default si no hay embeddings
    
    # Función interna para rellenar o truncar
    def pad_or_truncate(e_list, dim):
        if len(e_list) > dim: return e_list[:dim]
        if len(e_list) < dim: return e_list + [0.0] * (dim - len(e_list))
        return e_list
    
    processed_list = [pad_or_truncate(e, target_dim) for e in embeddings_list]
    return np.array(processed_list)

# --- Listas de Features (ACTUALIZADAS) ---
NUMERIC_FEATURES = [
    'life_cycle_length', 'num_stores', 'num_sizes', 'has_plus_sizes', 'price',
    'month_sin',     # <-- Feature de Fourier
    'month_cos',     # <-- Feature de Fourier
    'similarity_to_season_center',   # <-- Feature de Embedding
    'avg_distance_within_season',    # <-- Feature de Embedding
    'similarity_to_nearest_seasons'  # <-- Feature de Embedding
]
CATEGORICAL_FEATURES = [
    'id_season', 'aggregated_family', 'family', 'category', 'fabric', 
    'color_name', 'length_type', 'silhouette_type', 'waist_type', 
    'neck_lapel_type', 'sleeve_length_type', 'heel_shape_type', 
    'toecap_type', 'woven_structure', 'knit_structure', 'print_type', 
    'archetype', 'moment', 
    'embedding_cluster' # <-- Feature de Embedding
]
EMBEDDING_COLUMN = ['image_embedding'] # Columna original para PCA

print(f"Features Numéricas: {len(NUMERIC_FEATURES)}")
print(f"Features Categóricas: {len(CATEGORICAL_FEATURES)}")
print(f"Features de Embedding (PCA): 1")

# --- Pipeline Numérico (Común para ambos modelos) ---
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Pipeline de Embedding (Común para ambos modelos) ---
embedding_pipeline_pca = Pipeline(steps=[
    ('parser', FunctionTransformer(parse_embeddings)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=40, random_state=42)) # 50 componentes
])

# --- Pipeline 1: Preprocesador para LightGBM (con One-Hot Encoding) ---
categorical_pipeline_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 'most_frequent' o 'constant'
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline_ohe, CATEGORICAL_FEATURES),
        ('embed_pca', embedding_pipeline_pca, EMBEDDING_COLUMN)
    ],
    remainder='drop',
    n_jobs=-1
)

# --- Pipeline 2: Preprocesador para CatBoost (Manejo Nativo de Categóricas) ---
# CatBoost maneja NaNs nativamente, pero necesitamos convertir las categóricas a string
def convert_to_string(X):
    """Convierte todas las columnas a string y rellena NaNs."""
    X_copy = X.copy()
    for col in X_copy.columns:
        X_copy[col] = X_copy[col].fillna('_MISSING_').astype(str)
    return X_copy

categorical_pipeline_cat = Pipeline(steps=[
    ('to_string', FunctionTransformer(convert_to_string))
])

preprocessor_catboost = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline_cat, CATEGORICAL_FEATURES),
        ('embed_pca', embedding_pipeline_pca, EMBEDDING_COLUMN)
    ],
    remainder='drop',
    n_jobs=-1
)


# --- [PASO 6: Definición de Modelos] ---
print("--- Definiendo Modelos (LGBM y CatBoost) ---")

# Modelo 1: LightGBM (con tus hiperparámetros)
lgbm_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.70,          # Percentil 70
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=45,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# Pipeline completo para LGBM
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_lgbm),
    ('model', lgbm_model)
])

# Modelo 2: CatBoost
# Necesitamos decirle a CatBoost qué columnas son categóricas
# (después del preprocesador)
cat_feature_indices = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

catboost_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function='Quantile:alpha=0.70', # Mismo objetivo que LGBM
    eval_metric='MAE',
    random_seed=42,
    verbose=0,
    cat_features=cat_feature_indices, # <-- Clave para CatBoost
    # task_type="GPU", # Descomentar si tienes GPU y CatBoost con soporte
)

# Pipeline completo para CatBoost
catboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_catboost),
    ('model', catboost_model)
])


# --- [PASO 7: Entrenamiento Simple (Sin Validación Cruzada)] ---
print("\n" + "="*60)
print("ENTRENANDO MODELOS CON TODO EL CONJUNTO DE ENTRENAMIENTO")
print("="*60)

# --- Entrenar LightGBM ---
print("Entrenando LightGBM...")
lgbm_pipeline.fit(X, y)
print("LightGBM entrenado.")

# --- Entrenar CatBoost ---
print("Entrenando CatBoost...")
catboost_pipeline.fit(X, y)
print("CatBoost entrenado.")


# --- [PASO 8: Creación del Archivo de Submission Final] ---
print("\n" + "="*60)
print("GENERANDO PREDICCIONES FINALES (SOBRE TEST)")
print("="*60)

# Predecir sobre el conjunto de test
final_preds_lgbm = lgbm_pipeline.predict(X_test)
final_preds_catboost = catboost_pipeline.predict(X_test)

# Ensamble final (promedio de los dos modelos)
final_predictions = (final_preds_lgbm + final_preds_catboost) / 2

# Post-procesamiento final (asegurar que no haya demanda negativa)
final_predictions[final_predictions < 0] = 0

# Crear DataFrame de submission
submission_df = pd.DataFrame({
    'ID': test_ids_for_submission,
    'demand': final_predictions
})

# Guardar archivo
submission_filename = 'submission_GroupKFold_Ensemble_Fourier.csv'
submission_df.to_csv(submission_filename, index=False, sep=',')

print(f"¡Archivo '{submission_filename}' creado con éxito!")
print("\nVistazo al archivo de envío:")
print(submission_df.head(10))
print("\nEstadísticas de las predicciones:")
print(submission_df['demand'].describe())
print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)