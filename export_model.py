# --- Script para Entrenar y Exportar Modelo para Flutter ---
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import warnings

# Modelos
import lightgbm as lgb
from catboost import CatBoostRegressor

# Preprocessing y Pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

print("="*60)
print("EXPORTANDO MODELO PARA FLUTTER")
print("="*60)


def stable_string_to_unit(value: str) -> float:
    """Convierte una cadena en un número estable entre 0 y 1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = ''
    value = str(value)
    modulus = 1_000_003
    hash_value = 0
    for ch in value:
        hash_value = (hash_value * 131 + ord(ch)) % modulus
    return hash_value / modulus


KNN_NUMERIC_FEATURES = [
    'life_cycle_length', 'num_stores', 'num_sizes', 'has_plus_sizes', 'price',
    'month_sin', 'month_cos', 'similarity_to_season_center',
    'avg_distance_within_season', 'similarity_to_nearest_seasons', 'embedding_cluster'
]

KNN_CATEGORICAL_FEATURES = [
    'aggregated_family', 'family', 'category', 'fabric', 'color_name',
    'length_type', 'silhouette_type', 'waist_type', 'neck_lapel_type',
    'sleeve_length_type', 'heel_shape_type', 'toecap_type', 'woven_structure',
    'knit_structure', 'print_type', 'archetype', 'moment', 'season',
    'sleeve_length_type_X_season', 'family_X_season', 'fabric_X_season',
    'length_type_X_season'
]


def export_knn_dataset(features_df, target_series, output_path='knn_dataset.json', sample_size=8000):
    print("\n--- CREANDO DATASET PARA KNN ---")
    df = features_df.copy()
    df['__target__'] = target_series
    df = df[df['__target__'].notnull()]

    if df.empty:
        print("⚠ No hay datos disponibles para el dataset KNN")
        return

    rng = np.random.RandomState(42)
    sample_size = min(sample_size, len(df))
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)
    df_sample = df.iloc[sample_indices].reset_index(drop=True)

    def build_vector(row):
        vector = []
        for col in KNN_NUMERIC_FEATURES:
            value = row.get(col, 0)
            if pd.isna(value):
                value = 0
            vector.append(float(value))
        for col in KNN_CATEGORICAL_FEATURES:
            value = row.get(col, '')
            vector.append(stable_string_to_unit(value))
        return vector

    feature_vectors = df_sample.apply(build_vector, axis=1).tolist()
    matrix = np.array(feature_vectors, dtype=np.float32)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1e-6
    standardized_matrix = (matrix - means) / stds

    schema = (
        [{'name': col, 'type': 'numeric'} for col in KNN_NUMERIC_FEATURES] +
        [{'name': col, 'type': 'categorical'} for col in KNN_CATEGORICAL_FEATURES]
    )

    feature_stats = [
        {'mean': float(means[idx]), 'std': float(stds[idx])}
        for idx in range(len(schema))
    ]

    samples_payload = []
    targets = df_sample['__target__'].astype(float).tolist()
    for vec, target in zip(standardized_matrix.tolist(), targets):
        samples_payload.append({
            'features': [round(float(v), 6) for v in vec],
            'target': round(float(target), 4)
        })

    payload = {
        'schema': schema,
        'feature_stats': feature_stats,
        'samples': samples_payload,
        'k': 25,
        'sample_size': len(samples_payload)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"✓ Dataset KNN guardado en '{output_path}' con {len(samples_payload)} muestras")

# --- [PASO 1: Carga de Datos] ---
print("\n--- Cargando Datos ---")
train_df = pd.read_csv('train.csv', sep=';')
test_df = pd.read_csv('test.csv', sep=';')
print(f"Datos cargados: Train={train_df.shape}, Test={test_df.shape}")

# --- [PASO 2: Agregación de Datos] ---
print("--- Agregando Datos ---")
y_train_agg = train_df.groupby('ID')['weekly_demand'].sum()
X_train_agg_full = train_df.groupby('ID').first()

train_features = set(X_train_agg_full.columns)
test_features = set(test_df.columns)
common_columns = list(train_features.intersection(test_features))

X = X_train_agg_full[common_columns].copy()
X_test = test_df[common_columns].copy()
y = y_train_agg.reindex(X.index)

# --- [PASO 3: Features Temporales] ---
print("--- Creando features temporales ---")

def create_temporal_features(df):
    df_copy = df.copy()
    df_copy['phase_in'] = pd.to_datetime(df_copy['phase_in'], errors='coerce')
    df_copy['start_month'] = df_copy['phase_in'].dt.month
    median_month = 6.5
    df_copy['start_month'] = df_copy['start_month'].fillna(median_month)
    
    season_map = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
        12: 'Winter'
    }
    df_copy['season'] = df_copy['start_month'].round().astype(int).map(season_map)
    df_copy['season'] = df_copy['season'].fillna('Unknown')
    
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['start_month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['start_month'] / 12)
    df_copy = df_copy.drop('start_month', axis=1)
    
    return df_copy

X = create_temporal_features(X)
X_test = create_temporal_features(X_test)

# --- [PASO 4: Features de Embeddings] ---
print("--- Creando features de embeddings ---")

def create_embedding_features(df, embedding_col='image_embedding'):
    df = df.copy()
    
    if df['id_season'].isnull().any():
        df['id_season'] = df['id_season'].fillna(-1).astype(int)

    df['embedding_parsed'] = df[embedding_col].fillna('').apply(
        lambda x: np.array([float(v) for v in x.split(',') if v.strip()]) if x else np.zeros(256)
    )

    max_len = max(len(e) for e in df['embedding_parsed'])
    max_len = max(max_len, 256)
    
    df['embedding_parsed'] = df['embedding_parsed'].apply(
        lambda x: np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else x[:max_len]
    )

    embeddings_matrix = np.stack(df['embedding_parsed'].values)
    
    df['embedding_cluster'] = -1
    df['similarity_to_season_center'] = 0.0
    df['avg_distance_within_season'] = 0.0
    df['similarity_to_nearest_seasons'] = 0.0

    unique_seasons = df['id_season'].unique()
    
    # Clustering por temporada
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

    # Centroides y similitud
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

    # Distancias intra-temporada
    for season in unique_seasons:
        if season == -1: continue
        season_mask = (df['id_season'] == season)
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 1:
            distances = cdist(season_embeddings, season_embeddings, metric='euclidean')
            avg_dists = distances.sum(axis=1) / (len(season_embeddings) - 1)
            df.loc[season_mask, 'avg_distance_within_season'] = avg_dists

    # Similitud entre temporadas
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
    return df

X = create_embedding_features(X)
X_test = create_embedding_features(X_test)

# --- [PASO 4.5: Features de Tendencias] ---
print("--- Creando features de interacción ---")

def create_trend_features(df):
    df_copy = df.copy()
    
    if 'season' not in df_copy.columns:
        return df
    
    trend_attributes = ['sleeve_length_type', 'family', 'fabric', 'length_type']
    
    for attr in trend_attributes:
        if attr in df_copy.columns:
            attr_col = df_copy[attr].fillna('NA').astype(str)
            season_col = df_copy['season'].astype(str)
            df_copy[f'{attr}_X_season'] = attr_col + '_S_' + season_col
            
    return df_copy

X = create_trend_features(X)
X_test = create_trend_features(X_test)

# --- [PASO 5: Definir Features] ---
print("--- Definiendo features ---")

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
    'season', 'sleeve_length_type_X_season', 'family_X_season', 
    'fabric_X_season', 'length_type_X_season'
]

EMBEDDING_COLUMN = ['image_embedding']

# --- [PASO 6: Crear Pipelines] ---
print("--- Creando pipelines ---")

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

embedding_pipeline_pca = Pipeline(steps=[
    ('parser', FunctionTransformer(parse_embeddings)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=40, random_state=42))
])

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

lgbm_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.70,
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

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_lgbm),
    ('model', lgbm_model)
])

# --- [PASO 7: Entrenar Modelo] ---
print("\n--- ENTRENANDO MODELO LIGHTGBM ---")
lgbm_pipeline.fit(X, y)
print("✓ Modelo entrenado exitosamente")

# --- [PASO 8: Exportar Parámetros del Modelo] ---
print("\n--- EXPORTANDO PARÁMETROS ---")

# NO guardar el pipeline completo debido a problemas con pickle y FunctionTransformer
# Solo exportaremos los parámetros necesarios en JSON

# Exportar parámetros del preprocesador
preprocessor = lgbm_pipeline.named_steps['preprocessor']

# Extraer parámetros numéricos
num_transformer = preprocessor.named_transformers_['num']
num_imputer = num_transformer.named_steps['imputer']
num_scaler = num_transformer.named_steps['scaler']

numeric_params = {
    'features': NUMERIC_FEATURES,
    'imputer_strategy': num_imputer.strategy,
    'imputer_statistics': num_imputer.statistics_.tolist() if num_imputer.statistics_ is not None else None,
    'scaler_mean': num_scaler.mean_.tolist() if num_scaler.mean_ is not None else None,
    'scaler_scale': num_scaler.scale_.tolist() if num_scaler.scale_ is not None else None
}

# Extraer parámetros categóricos
cat_transformer = preprocessor.named_transformers_['cat']
cat_imputer = cat_transformer.named_steps['imputer']
cat_onehot = cat_transformer.named_steps['onehot']

categorical_params = {
    'features': CATEGORICAL_FEATURES,
    'imputer_strategy': cat_imputer.strategy,
    'imputer_fill_value': str(cat_imputer.fill_value) if hasattr(cat_imputer, 'fill_value') else None,
    'onehot_categories': [cat.tolist() for cat in cat_onehot.categories_],
    'onehot_feature_names': cat_onehot.get_feature_names_out().tolist()
}

# Extraer parámetros de PCA
embed_transformer = preprocessor.named_transformers_['embed_pca']
pca = embed_transformer.named_steps['pca']
embed_scaler = embed_transformer.named_steps['scaler']

embedding_params = {
    'n_components': pca.n_components,
    'pca_mean': pca.mean_.tolist(),
    'pca_components': pca.components_.tolist(),
    'pca_explained_variance': pca.explained_variance_.tolist(),
    'scaler_mean': embed_scaler.mean_.tolist() if embed_scaler.mean_ is not None else None,
    'scaler_scale': embed_scaler.scale_.tolist() if embed_scaler.scale_ is not None else None
}

# Exportar parámetros del modelo LightGBM
model = lgbm_pipeline.named_steps['model']
model_params = {
    'model_type': 'LGBMRegressor',
    'objective': 'quantile',
    'alpha': 0.70,
    'n_estimators': model.n_estimators,
    'learning_rate': model.learning_rate,
    'num_leaves': model.num_leaves,
    'max_depth': model.max_depth,
}

# Guardar el modelo en formato JSON de LightGBM
model.booster_.save_model('lgbm_model.txt')
print("✓ Modelo LightGBM guardado en 'lgbm_model.txt'")

# Crear diccionario completo de exportación
export_data = {
    'numeric_preprocessing': numeric_params,
    'categorical_preprocessing': categorical_params,
    'embedding_preprocessing': embedding_params,
    'model_params': model_params,
    'feature_names': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    'embedding_column': EMBEDDING_COLUMN[0]
}

# Guardar todo en JSON
with open('model_params.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)
print("✓ Parámetros guardados en 'model_params.json'")

# Exportar dataset para KNN
export_knn_dataset(X, y, output_path='knn_dataset.json')


# --- [PASO 9: Estadísticas del Modelo] ---
print("\n" + "="*60)
print("ESTADÍSTICAS DEL MODELO")
print("="*60)

# Hacer predicciones en train para ver el rendimiento
train_preds = lgbm_pipeline.predict(X)
train_residuals = y - train_preds

print(f"\nFeatures numéricas: {len(NUMERIC_FEATURES)}")
print(f"Features categóricas: {len(CATEGORICAL_FEATURES)}")
print(f"Total features después de preprocessing: {preprocessor.transform(X[:1]).shape[1]}")
print(f"\nEstadísticas de predicciones en Train:")
print(f"  Media: {train_preds.mean():.2f}")
print(f"  Mediana: {np.median(train_preds):.2f}")
print(f"  Min: {train_preds.min():.2f}")
print(f"  Max: {train_preds.max():.2f}")
print(f"\nTarget Real (Train):")
print(f"  Media: {y.mean():.2f}")
print(f"  Mediana: {y.median():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

print("\n" + "="*60)
print("EXPORTACIÓN COMPLETADA")
print("="*60)
print("\nArchivos generados:")
print("  1. lgbm_model.txt - Modelo LightGBM en formato texto")
print("  2. model_params.json - Parámetros para Flutter/Dart")
print("\n✓ Listo para integrar en Flutter")
