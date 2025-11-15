# Importació de llibreries bàsiques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast  # Per processar els embeddings
from pathlib import Path

# Preprocessament i Modelització
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# Càrrega de dades
data_dir = Path('.')
train_path = data_dir / 'train.csv'
test_path = data_dir / 'test.csv'
sample_path = data_dir / 'sample_submission.csv'

try:
    train_df = pd.read_csv(train_path, sep=';')
    test_df = pd.read_csv(test_path, sep=';')
    sample_sub = pd.read_csv(sample_path, sep=',')
except FileNotFoundError:
    print("Error: Assegura't que els fitxers 'train.csv', 'test.csv' i 'sample_submission.csv' estan al mateix directori.")

print("Dades carregades:")
print(f"Train:   {train_df.shape}")
print(f"Test:    {test_df.shape}")
print(f"Sample:  {sample_sub.shape}")

# --- Paso 1: Tarea 2 (Inspección Rápida) ---

print("--- Vistazo a train_df (head) ---")
print(train_df.head())
print("\n")

print("--- Información de train_df (.info()) ---")
# Esto nos dirá los tipos de datos (Dtypes) y los nulos
train_df.info()
print("\n")

print("--- Información de test_df (.info()) ---")
test_df.info()
print("\n")

# --- Paso 1: Tarea 3 (Validar Hipótesis de Agregación) ---

print("--- Conteo de filas por 'id' en train_df ---")
# Si vemos números mayores a 1, significa que hay varias filas por id (semanal)
print(train_df['ID'].value_counts().head(10))
print("\n")

print("--- Conteo de filas por 'id' en test_df ---")
# Aquí deberíamos ver que todos los valores son '1'
print(test_df['ID'].value_counts().head(10))
print("\n")

# --- [INICIO] Código Corregido para Paso 2 ---

# --- Paso 2: Tarea 1 (Crear el Target 'y_train_agg') ---
print("Agregando 'weekly_demand' para crear el target (y_train_agg)...")

# (CORRECCIÓN): Usamos 'ID' (mayúscula) como vimos en tu CSV
y_train_agg = train_df.groupby('ID')['weekly_demand'].sum()

print("Target (y_train_agg) creado. Ejemplo:")
print(y_train_agg.head())
print("\n")


# --- Paso 2: Tarea 2 (Crear las Features 'X_train_agg') ---
print("Agregando features (X_train_agg)...")

# (CORRECCIÓN): Usamos 'ID' (mayúscula)
X_train_agg_full = train_df.groupby('ID').first()


# --- Tarea 2b: Alinear Columnas (¡NUEVA LÓGICA MÁS ROBUSTA!) ---
# En lugar de usar la lista 'test_df.columns' (que tiene 'ID' y 'Unnamed'),
# vamos a buscar las columnas que SÍ están en ambos sitios.

train_features = set(X_train_agg_full.columns)
test_features = set(test_df.columns)

# Buscamos la intersección: columnas que están en X_train_agg_full Y en test_df
# Esto excluye automáticamente 'ID' (que no es columna en train) 
# y 'Unnamed:' (que no están en train)
common_columns = list(train_features.intersection(test_features))

print(f"Encontradas {len(common_columns)} columnas en común.")

# Ahora filtramos ambos DataFrames para que SÓLO tengan estas columnas
X_train_agg = X_train_agg_full[common_columns].copy() # Usamos .copy() para evitar warnings

# ¡Importante! También limpiamos test_df para que coincida
# (Guardamos el 'ID' original de test_df para la sumisión final)
test_ids_for_submission = test_df['ID']
test_df_clean = test_df[common_columns].copy() 


# --- Tarea 2c: Alineación Final ---
# Nos aseguramos de que 'y_train_agg' siga el mismo orden que 'X_train_agg'
y_train_agg = y_train_agg.reindex(X_train_agg.index)

print("\n¡Agregación completada y alineada con éxito!")
print(f"Forma de X_train_agg: {X_train_agg.shape}")
print(f"Forma de y_train_agg: {y_train_agg.shape}")
print(f"Forma de test_df_clean: {test_df_clean.shape}")
print("\n")

print("--- Columnas de X_train_agg (.info()) ---")
X_train_agg.info()
print("\n")
print("--- Columnas de test_df_clean (.info()) ---")
test_df_clean.info()

# --- [FIN] Código Corregido para Paso 2 ---

# --- Paso 3: Ingeniería de Características ---

print("Iniciando Paso 3: Ingeniería de Características...")

# Hacemos una copia para evitar 'SettingWithCopyWarning'
X_train_features = X_train_agg.copy()
X_test_features = test_df_clean.copy()

# --- Tarea 1: Crear 'start_month' (Ingeniería de Fechas) ---
# Convertimos 'phase_in' a formato fecha (MANDATORIO para extraer el mes)
# errors='coerce' convertirá cualquier fecha inválida en NaT (Nulo)
X_train_features['phase_in'] = pd.to_datetime(X_train_features['phase_in'], errors='coerce')
X_test_features['phase_in'] = pd.to_datetime(X_test_features['phase_in'], errors='coerce')

# Extraemos el mes (1-12) como una nueva feature numérica
X_train_features['start_month'] = X_train_features['phase_in'].dt.month
X_test_features['start_month'] = X_test_features['phase_in'].dt.month

# Ahora rellenamos cualquier nulo que se haya creado (por si 'errors=coerce' falló)
# Usaremos la mediana (ej. 6, para Junio/Julio)
median_month = X_train_features['start_month'].median()
X_train_features['start_month'] = X_train_features['start_month'].fillna(median_month)
X_test_features['start_month'] = X_test_features['start_month'].fillna(median_month)

print("Feature 'start_month' creada con éxito. Ejemplo:")
print(X_train_features[['phase_in', 'start_month']].head())
print("\n")


# --- Tarea 2: Definir Listas de Columnas para el Pipeline ---
# Basado en nuestro análisis del .info()

# 1. Columnas Numéricas: Rellenaremos nulos con la mediana y las escalaremos
numeric_features = [
    'num_stores',
    'price',
    'life_cycle_length',
    'num_sizes',
    'start_month'  # Nuestra nueva feature
]

# 2. Columnas Categóricas: Rellenaremos nulos con "Desconocido" y aplicaremos One-Hot Encoding
categorical_features = [
    'moment',
    'archetype',
    'neck_lapel_type',
    'print_type',
    'has_plus_sizes',       # Aunque es bool, lo tratamos como categórico
    'knit_structure',
    'waist_type',
    'silhouette_type',
    'family',
    'length_type',
    'color_name',
    'category',
    'woven_structure',
    'id_season',            # Aunque es int, es una CATEGORÍA (ej. temporada 86)
    'aggregated_family',
    'sleeve_length_type',
    'fabric'
]

# 3. Columnas a Eliminar: No las usaremos
# (El Pipeline se encargará de esto automáticamente al NO incluirlas 
# en las listas de arriba, excepto las que creamos y ya no necesitamos)

print(f"Definidas {len(numeric_features)} features numéricas:")
print(numeric_features)
print("\n")
print(f"Definidas {len(categorical_features)} features categóricas:")
print(categorical_features)
print("\n¡Paso 3 completado! Estamos listos para construir el Pipeline.")

# --- Paso 4: Preprocesamiento y Pipeline de Scikit-learn ---

print("Herramientas de Scikit-learn importadas.")

# --- Tarea 1: Definir los Mini-Pipelines ---

# Pipeline para datos NUMÉRICOS
# 1. SimpleImputer: Rellena cualquier nulo (ej. en 'start_month') con la mediana.
# 2. StandardScaler: Escala los datos (pone todo ~ entre -2 y 2). 
#    RandomForest no lo necesita, pero es BUENA PRÁCTICA para otros modelos.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para datos CATEGÓRICOS
# 1. SimpleImputer: Rellena los nulos (ej. en 'archetype') con la palabra "Desconocido".
# 2. OneHotEncoder: Crea nuevas columnas para cada categoría (ej. family_Dresses, family_Coats).
#    handle_unknown='ignore' -> Si en test_df aparece un color que no vio en train,
#    simplemente lo ignora sin dar error. ¡Vital!
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

print("Pipelines de transformación numérica y categórica definidos.")

# --- Tarea 2: Unir con ColumnTransformer ---
# Aquí es donde le decimos a sklearn:
# - Aplica 'numeric_transformer' a las columnas de 'numeric_features'
# - Aplica 'categorical_transformer' a las columnas de 'categorical_features'
# - remainder='drop' -> Todas las columnas que no mencionamos (ej. image_embedding) serán ELIMINADAS.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Ignora las columnas que no están en nuestras listas
)

print("ColumnTransformer ('preprocessor') creado con éxito.")
print("\n¡Paso 4 completado! El 'preprocessor' está listo para ser usado.")

# --- Bonus: Verifiquemos qué ha creado ---
# Vamos a "entrenar" solo el preprocesador y ver la forma de la salida
# Usamos las variables que creamos en el Paso 3

# Renombramos por simplicidad
X = X_train_features
y = y_train_agg
X_test = X_test_features

# 'fit_transform' aprende de X (ej. calcula la mediana) y luego transforma X
X_processed = preprocessor.fit_transform(X)

print("\n--- Verificación del Preprocesador ---")
print(f"Forma original de X_train: {X.shape}")
print(f"Forma de X_train procesado: {X_processed.shape}")
print("¡El número de columnas ha crecido por el OneHotEncoding!")

# --- Paso 5: Entrenamiento del Modelo ---
from sklearn.ensemble import RandomForestRegressor

print("Herramientas de modelo y métricas importadas.")

# --- Tarea 1: Separar Datos de Validación ---
# Dividimos nuestros datos (X, y) en 80% para entrenar y 20% para validar
# random_state=42 asegura que la división sea siempre la misma
X_train_local, X_val_local, y_train_local, y_val_local = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Datos divididos: {len(X_train_local)} para entrenar, {len(X_val_local)} para validar.")
print("\n")

# --- Tarea 2: Crear el Pipeline Final ---
# Unimos el 'preprocessor' (Paso 4) y nuestro modelo 'RandomForestRegressor'

# n_estimators=100 -> Crear un bosque de 100 árboles (es un buen número para empezar)
# n_jobs=-1 -> Usar todos los núcleos de tu CPU para entrenar más rápido
# random_state=42 -> Para que el entrenamiento sea reproducible
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("Pipeline final creado (Preprocesador + RandomForest).")

# --- Tarea 3: Entrenar ---
print("Entrenando el modelo... (Esto puede tardar 1-2 minutos)")
# ¡Aquí ocurre la magia!
# .fit() entrena el preprocesador y el modelo, todo en una línea.
full_pipeline.fit(X_train_local, y_train_local)

print("¡Modelo entrenado con éxito!")
print("\n")


# --- Paso 6: Evaluación y Ajuste ---

print("--- Evaluación del Modelo (sobre datos de validación) ---")

# 1. Predecir sobre los datos de validación
y_pred_val = full_pipeline.predict(X_val_local)

# 2. Calcular Métricas de Error
mae = mean_absolute_error(y_val_local, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val_local, y_pred_val))

print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print("\n")

print("--- Interpretación ---")
print(f"Un MAE de {mae:.4f} significa que, en promedio, las predicciones de demanda total del modelo")
print(f"tienen un error de +/- {mae:.4f} unidades (en la escala 0-1).")
print("\n¡Paso 5 y 6 completados!")

import lightgbm as lgb
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# --- PASO 1: Función para procesar los Embeddings ---
# Esta función toma el DataFrame de una columna ('image_embedding')
# y convierte cada string en un vector numérico.
def parse_embeddings(df_column):
    """
    Toma una columna de un DataFrame (pasada por ColumnTransformer) que contiene
    strings de embeddings separados por comas y los convierte en una matriz 2D de NumPy.
    """
    embeddings_series = df_column.iloc[:, 0].fillna('')

    embeddings_list = embeddings_series.apply(
        lambda x: [float(v) for v in x.split(',') if v.strip()] if x else []
    )

    try:
        target_dim = len(next(item for item in embeddings_list if item))
    except StopIteration:
        target_dim = 256
        print("Advertencia: No se encontraron embeddings válidos, usando dimensión 256 por defecto.")

    def pad_or_truncate(e_list, dim):
        if len(e_list) > dim:
            return e_list[:dim]
        if len(e_list) < dim:
            return e_list + [0.0] * (dim - len(e_list))
        return e_list

    processed_list = [pad_or_truncate(e, target_dim) for e in embeddings_list]
    return np.array(processed_list)


def create_embedding_features(df, embedding_col='image_embedding'):
    """Crea features basadas en embeddings y temporada.

    - Clustering por temporada (KMeans)
    - Similitud con el centroide de la temporada
    - Distancia media a otros productos de la misma temporada
    - Similitud media con temporadas más cercanas
    """
    print("Procesando embeddings para crear features estacionales...")

    df = df.copy()
    df['embedding_parsed'] = df[embedding_col].fillna('').apply(
        lambda x: np.array([float(v) for v in x.split(',') if v.strip()]) if x else np.zeros(256)
    )

    max_len = max(len(e) for e in df['embedding_parsed'])
    df['embedding_parsed'] = df['embedding_parsed'].apply(
        lambda x: np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else x[:max_len]
    )

    embeddings_matrix = np.stack(df['embedding_parsed'].values)

    print("Aplicando clustering por temporada...")
    df['embedding_cluster'] = -1
    season_clusters = {}

    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            if len(season_embeddings) > 5:
                n_clusters = min(5, len(season_embeddings) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(season_embeddings)
                df.loc[season_mask, 'embedding_cluster'] = clusters
                season_clusters[season] = kmeans.cluster_centers_

    print("Calculando centroides por temporada...")
    season_centroids = {}
    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            if len(season_embeddings) > 0:
                season_centroids[season] = season_embeddings.mean(axis=0)

    df['similarity_to_season_center'] = 0.0
    for season, centroid in season_centroids.items():
        season_mask = df['id_season'] == season
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 0:
            sims = cosine_similarity(season_embeddings, centroid.reshape(1, -1)).flatten()
            df.loc[season_mask, 'similarity_to_season_center'] = sims

    print("Calculando distancias intra-temporada...")
    df['avg_distance_within_season'] = 0.0
    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            if len(season_embeddings) > 1:
                distances = cdist(season_embeddings, season_embeddings, metric='euclidean')
                avg_dists = distances.sum(axis=1) / (len(season_embeddings) - 1)
                df.loc[season_mask, 'avg_distance_within_season'] = avg_dists

    print("Calculando similitudes entre temporadas...")
    df['similarity_to_nearest_seasons'] = 0.0
    if len(season_centroids) > 1:
        centroid_matrix = np.array(list(season_centroids.values()))
        for idx, (season, centroid) in enumerate(season_centroids.items()):
            season_mask = df['id_season'] == season
            other_centroids = np.delete(centroid_matrix, idx, axis=0)
            sims = cosine_similarity(centroid.reshape(1, -1), other_centroids).flatten()
            top_sims = np.sort(sims)[-3:] if len(sims) >= 3 else sims
            df.loc[season_mask, 'similarity_to_nearest_seasons'] = top_sims.mean()

    df = df.drop('embedding_parsed', axis=1)
    print("Features de embeddings estacionales creadas!")
    return df


# --- PASO EXTRA: Aplicar las features estacionales sobre los agregados ---
print("\nAplicando features de estacionalidad basadas en embeddings...")
X = create_embedding_features(X_train_features)
X_test = create_embedding_features(X_test_features)

print("Formas tras añadir features de estacionalidad:")
print(f"X (train): {X.shape}")
print(f"X_test:    {X_test.shape}")
print("\n")

# --- PASO 7: Actualizar Listas de Features con las nuevas columnas estacionales ---
print("Actualizando listas de features para incluir features estacionales...")

# Columnas numéricas: añadimos las nuevas features de embeddings
NUMERICAL_FEATURES = [
    'life_cycle_length', 'num_stores', 'num_sizes', 'has_plus_sizes', 'price',
    'start_month',  # Feature de ingeniería
    'similarity_to_season_center',  # Nueva feature estacional
    'avg_distance_within_season',   # Nueva feature estacional
    'similarity_to_nearest_seasons'  # Nueva feature estacional
]

# Columnas categóricas: añadimos embedding_cluster
CATEGORICAL_FEATURES = [
    'id_season', 'aggregated_family', 'family', 'category', 'fabric', 
    'color_name', 'length_type', 'silhouette_type', 'waist_type', 
    'neck_lapel_type', 'sleeve_length_type', 'heel_shape_type', 
    'toecap_type', 'woven_structure', 'knit_structure', 'print_type', 
    'archetype', 'moment', 
    'embedding_cluster'  # Nueva feature estacional
]

# La columna de embedding raw para PCA
EMBEDDING_COLUMN = ['image_embedding']

print(f"Features numéricas ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
print(f"Features categóricas ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
print("\n")

# --- PASO 8: Crear pipelines mejorados ---
print("Creando pipelines de preprocesamiento mejorados...")

# Pipeline para datos numéricos
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para datos categóricos
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Pipeline para los embeddings: parser + PCA
embedding_pipeline = Pipeline(steps=[
    ('parser', FunctionTransformer(parse_embeddings)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=42))
])

# ColumnTransformer mejorado
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ('embed', embedding_pipeline, EMBEDDING_COLUMN)
    ],
    remainder='drop'
)

print("Preprocessor avanzado creado.")
print("\n")

# --- PASO 9: Crear modelo LightGBM con regresión de cuantiles ---
print("Configurando modelo LightGBM con regresión de cuantiles...")

lgbm_model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.70,  # Percentil 70
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

# Pipeline final avanzado
full_pipeline_advanced = Pipeline(steps=[
    ('preprocessor', preprocessor_advanced),
    ('model', lgbm_model)
])

print("Pipeline completo creado (Preprocessor + LightGBM Quantile).")
print("\n")

# --- PASO 10: Entrenar con todos los datos ---
print("="*60)
print("ENTRENAMIENTO FINAL CON TODOS LOS DATOS")
print("="*60)
print("Entrenando modelo con el 100% de los datos de entrenamiento...")

full_pipeline_advanced.fit(X, y)

print("¡Modelo entrenado con éxito!")
print("\n")

# --- PASO 11: Generar predicciones finales ---
print("Generando predicciones sobre datos de test...")

final_predictions = full_pipeline_advanced.predict(X_test)

print(f"Predicciones generadas: {len(final_predictions)} productos")
print(f"Rango de predicciones: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
print("\n")

# --- PASO 12: Post-procesamiento ---
print("Aplicando post-procesamiento...")

# Ajustar predicciones negativas a 0
final_predictions[final_predictions < 0] = 0
print("Predicciones negativas ajustadas a 0.")
print(f"Rango final: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
print("\n")

# --- PASO 13: Crear archivo de submission ---
print("="*60)
print("CREANDO ARCHIVO DE SUBMISSION")
print("="*60)

submission_df = pd.DataFrame({
    'ID': test_ids_for_submission,
    'demand': final_predictions
})

# Nombre del archivo
submission_filename = 'submission_EDA3_complete.csv'
submission_df.to_csv(submission_filename, index=False, sep=',')

print(f"¡Archivo '{submission_filename}' creado con éxito!")
print("\nVistazo al archivo de envío:")
print(submission_df.head(10))
print("\nEstadísticas de las predicciones:")
print(submission_df['demand'].describe())
print("\n")

print("="*60)
print("PROCESO COMPLETADO")
print("="*60)
print("\nMejoras implementadas en este modelo:")
print("✓ Clustering de embeddings por temporada")
print("✓ Similitud con centroide de temporada")
print("✓ Distancias intra-temporada")
print("✓ Similitud con temporadas cercanas")
print("✓ PCA sobre embeddings originales (50 componentes)")
print("✓ Modelo LightGBM con regresión de cuantiles (p70)")
print("✓ Regularización L1/L2 para evitar overfitting")
print(f"\n📁 Archivo listo para enviar: {submission_filename}")
print("="*60)

