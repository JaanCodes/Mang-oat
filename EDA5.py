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

# --- [PASO 2: Agregación de Datos CONSERVADORA] ---
print("--- Agregando Datos (groupby ID) ---")
# Target
y_train_agg = train_df.groupby('ID')['weekly_demand'].sum()

# Features: usar .first() para atributos de producto
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
print("--- Creando features de Fourier y Temporada ('season') ---")

def create_temporal_features(df):
    """Crea features de mes, temporada (Winter/Summer...) y sin/cos."""
    df_copy = df.copy()
    
    # 1. Crear 'start_month' (1-12)
    df_copy['phase_in'] = pd.to_datetime(df_copy['phase_in'], errors='coerce')
    df_copy['start_month'] = df_copy['phase_in'].dt.month
    
    # 2. Rellenar nulos del mes
    median_month = 6.5 # (entre junio y julio)
    df_copy['start_month'] = df_copy['start_month'].fillna(median_month)

    # 3. Crear feature 'season'
    season_map = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
        12: 'Winter'
    }
    df_copy['season'] = df_copy['start_month'].round().astype(int).map(season_map)
    df_copy['season'] = df_copy['season'].fillna('Unknown')
    
    # 4. Crear Features de Fourier
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['start_month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['start_month'] / 12)
    
    # 5. Eliminar la columna original
    df_copy = df_copy.drop('start_month', axis=1)
    
    return df_copy

X = create_temporal_features(X)
X_test = create_temporal_features(X_test)
print("Features 'month_sin', 'month_cos' y 'season' creadas.")


# --- [PASO 4: Ingeniería de Embeddings Estacionales] ---
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
    max_len = max(max_len, 256)
    
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
        if season == -1: continue
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 5:
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
        if len(season_embeddings) > 1:
            distances = cdist(season_embeddings, season_embeddings, metric='euclidean')
            avg_dists = distances.sum(axis=1) / (len(season_embeddings) - 1)
            df.loc[season_mask, 'avg_distance_within_season'] = avg_dists

    # --- Similitud entre temporadas ---
    if len(season_centroids) > 1:
        centroid_matrix = np.array(list(season_centroids.values()))
        
        for idx, (season, centroid) in enumerate(season_centroids.items()):
            season_mask = (df['id_season'] == season)
            other_centroids = np.delete(centroid_matrix, idx, axis=0)
            if other_centroids.shape[0] > 0:
                sims = cosine_similarity(centroid.reshape(1, -1), other_centroids).flatten()
                top_sims = np.sort(sims)[-3:] if len(sims) >= 3 else sims
                df.loc[season_mask, 'similarity_to_nearest_seasons'] = top_sims.mean()

    df = df.drop('embedding_parsed', axis=1)
    print("Features de embeddings estacionales creadas.")
    return df

X = create_embedding_features(X)
X_test = create_embedding_features(X_test)


# --- [PASO 4.5: Creación de Features de Interacción (Tendencias)] ---
print("--- Creando features de Interacción (Tendencias) ---")

def create_trend_features(df):
    """Crea features de interacción entre atributos clave y la temporada."""
    df_copy = df.copy()
    
    if 'season' not in df_copy.columns:
        print("Error: La columna 'season' no se encontró.")
        return df
    
    trend_attributes = ['sleeve_length_type', 'family', 'fabric', 'length_type']
    
    for attr in trend_attributes:
        if attr in df_copy.columns:
            attr_col = df_copy[attr].fillna('NA').astype(str)
            season_col = df_copy['season'].astype(str)
            df_copy[f'{attr}_X_season'] = attr_col + '_S_' + season_col
        else:
            print(f"Info: Columna '{attr}' no encontrada, se omite interacción.")
            
    return df_copy

X = create_trend_features(X)
X_test = create_trend_features(X_test)
print("Features de interacción creadas.")


# --- [PASO 5: Definición de Features y Pipelines] ---
print("--- Definiendo listas de Features y Pipelines ---")

# Función para parsear embeddings (para PCA)
def parse_embeddings(df_column):
    embeddings_series = df_column.iloc[:, 0].fillna('')
    embeddings_list = embeddings_series.apply(
        lambda x: [float(v) for v in x.split(',') if v.strip()] if x else []
    )
    try:
        target_dim = len(next(item for item in embeddings_list if item))
    except StopIteration:
        target_dim = 256
    
    def pad_or_truncate(e_list, dim):
        if len(e_list) > dim: return e_list[:dim]
        if len(e_list) < dim: return e_list + [0.0] * (dim - len(e_list))
        return e_list
    
    processed_list = [pad_or_truncate(e, target_dim) for e in embeddings_list]
    return np.array(processed_list)

# --- Listas de Features ---
NUMERIC_FEATURES = [
    'life_cycle_length', 'num_stores', 'num_sizes', 'has_plus_sizes', 'price',
    'month_sin', 'month_cos',
    'similarity_to_season_center', 'avg_distance_within_season', 'similarity_to_nearest_seasons'
]
CATEGORICAL_FEATURES = [
    'id_season', 'aggregated_family', 'family', 'category', 'fabric', 
    'color_name', 'length_type', 'silhouette_type', 'waist_type', 
    'neck_lapel_type', 'sleeve_length_type', 'heel_shape_type', 
    'toecap_type', 'woven_structure', 'knit_structure', 'print_type', 
    'archetype', 'moment', 'embedding_cluster',
    'season',
    'sleeve_length_type_X_season', 'family_X_season', 'fabric_X_season', 'length_type_X_season'
]
EMBEDDING_COLUMN = ['image_embedding']

print(f"Features Numéricas: {len(NUMERIC_FEATURES)}")
print(f"Features Categóricas: {len(CATEGORICAL_FEATURES)}")

# --- Pipelines ---
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

embedding_pipeline_pca = Pipeline(steps=[
    ('parser', FunctionTransformer(parse_embeddings)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=40, random_state=42))
])

# Pipeline para LightGBM (One-Hot Encoding)
categorical_pipeline_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
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

# Pipeline para CatBoost
def convert_to_string(X):
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


# --- [PASO 6: Definición de Modelos OPTIMIZADOS] ---
print("--- Definiendo Modelos (LGBM y CatBoost) ---")

# Modelo 1: LightGBM (Optimizado para minimizar ventas perdidas)
lgbm_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.75,          # Percentil 75 (balance entre exceso y ventas perdidas)
    n_estimators=2500,
    learning_rate=0.02,
    num_leaves=55,
    max_depth=10,
    min_child_samples=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.01,
    reg_lambda=0.01,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_lgbm),
    ('model', lgbm_model)
])

# Modelo 2: CatBoost
cat_feature_indices = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

catboost_model = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.02,
    depth=10,
    l2_leaf_reg=1.5,
    loss_function='Quantile:alpha=0.75',
    eval_metric='MAE',
    random_seed=42,
    verbose=0,
    cat_features=cat_feature_indices,
)

catboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_catboost),
    ('model', catboost_model)
])


# --- [PASO 7: Entrenamiento] ---
print("\n" + "="*60)
print("ENTRENANDO MODELOS")
print("="*60)

print("Entrenando LightGBM...")
lgbm_pipeline.fit(X, y)
print("LightGBM entrenado.")

print("Entrenando CatBoost...")
catboost_pipeline.fit(X, y)
print("CatBoost entrenado.")


# --- [PASO 8: Predicciones y Submission] ---
print("\n" + "="*60)
print("GENERANDO PREDICCIONES FINALES")
print("="*60)

# Predecir
final_preds_lgbm = lgbm_pipeline.predict(X_test)
final_preds_catboost = catboost_pipeline.predict(X_test)

# Ensamble (60% LGBM, 40% CatBoost - LGBM suele ser mejor en este tipo de datos)
final_predictions = (final_preds_lgbm * 0.6 + final_preds_catboost * 0.4)

# Post-procesamiento
final_predictions[final_predictions < 0] = 0

# Ajuste conservador: +5% para compensar ventas perdidas
final_predictions = final_predictions * 1.05

# Crear submission
submission_df = pd.DataFrame({
    'ID': test_ids_for_submission,
    'Production': final_predictions
})

submission_filename = 'submission_optimized_v3.csv'
submission_df.to_csv(submission_filename, index=False, sep=',')

print(f"¡Archivo '{submission_filename}' creado con éxito!")
print("\nVistazo al archivo de envío:")
print(submission_df.head(10))
print("\nEstadísticas de las predicciones:")
print(submission_df['Production'].describe())
print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)
