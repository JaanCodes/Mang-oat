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
    # df_column.iloc[:, 0] selecciona la primera (y única) columna como una Serie
    embeddings_series = df_column.iloc[:, 0]
    
    # Manejar posibles valores nulos o vacíos
    embeddings_series = embeddings_series.fillna('') 
    
    # Procesa cada string
    embeddings_list = embeddings_series.apply(
        lambda x: [float(v) for v in x.split(',') if v.strip()] if x else []
    )
    
    # Encontrar la longitud máxima esperada (basado en un valor no nulo)
    try:
        # Intenta inferir la longitud de un embedding válido
        target_dim = len(next(item for item in embeddings_list if item))
    except StopIteration:
        # Si no hay embeddings, usa una dimensión por defecto (ej. 256)
        target_dim = 256 
        print("Advertencia: No se encontraron embeddings válidos, usando default.")

    # Asegura que todos los vectores tengan la misma longitud, rellenando con 0 si es necesario
    def pad_or_truncate(e_list, dim):
        if len(e_list) > dim:
            return e_list[:dim]
        elif len(e_list) < dim:
            return e_list + [0.0] * (dim - len(e_list))
        return e_list

    processed_list = [pad_or_truncate(e, target_dim) for e in embeddings_list]
    
    # Devuelve la matriz NumPy
    return np.array(processed_list)


# --- NUEVO: Función para crear features avanzadas con embeddings ---
def create_embedding_features(df, embedding_col='image_embedding'):
    """
    Crea features basadas en embeddings:
    - Clustering por temporada
    - Similitud con centros de temporada
    - Estadísticas de distancias
    """
    print("Procesando embeddings para crear features avanzadas...")
    
    # Parsear embeddings
    df = df.copy()
    df['embedding_parsed'] = df[embedding_col].fillna('').apply(
        lambda x: np.array([float(v) for v in x.split(',') if v.strip()]) if x else np.zeros(256)
    )
    
    # Normalizar longitud de embeddings
    max_len = max(len(e) for e in df['embedding_parsed'])
    df['embedding_parsed'] = df['embedding_parsed'].apply(
        lambda x: np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else x[:max_len]
    )
    
    # Convertir a matriz
    embeddings_matrix = np.stack(df['embedding_parsed'].values)
    
    # 1. Clustering de embeddings por temporada usando KMeans
    print("Aplicando clustering por temporada...")
    season_clusters = {}
    df['embedding_cluster'] = -1
    
    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            
            if len(season_embeddings) > 5:  # Solo si hay suficientes muestras
                n_clusters = min(5, len(season_embeddings) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(season_embeddings)
                df.loc[season_mask, 'embedding_cluster'] = clusters
                season_clusters[season] = kmeans.cluster_centers_
    
    # 2. Calcular centroide por temporada
    print("Calculando centroides por temporada...")
    season_centroids = {}
    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            if len(season_embeddings) > 0:
                season_centroids[season] = season_embeddings.mean(axis=0)
    
    # 3. Calcular similitud con centroide de su temporada
    df['similarity_to_season_center'] = 0.0
    for season, centroid in season_centroids.items():
        season_mask = df['id_season'] == season
        season_embeddings = embeddings_matrix[season_mask]
        if len(season_embeddings) > 0:
            similarities = cosine_similarity(season_embeddings, centroid.reshape(1, -1)).flatten()
            df.loc[season_mask, 'similarity_to_season_center'] = similarities
    
    # 4. Distancia media a otros productos de la misma temporada
    print("Calculando distancias intra-temporada...")
    df['avg_distance_within_season'] = 0.0
    for season in df['id_season'].unique():
        if pd.notna(season):
            season_mask = df['id_season'] == season
            season_embeddings = embeddings_matrix[season_mask]
            if len(season_embeddings) > 1:
                distances = cdist(season_embeddings, season_embeddings, metric='euclidean')
                avg_distances = distances.sum(axis=1) / (len(season_embeddings) - 1)
                df.loc[season_mask, 'avg_distance_within_season'] = avg_distances
    
    # 5. Similitud con las 3 temporadas más cercanas
    print("Calculando similitudes entre temporadas...")
    df['similarity_to_nearest_seasons'] = 0.0
    if len(season_centroids) > 1:
        centroid_matrix = np.array(list(season_centroids.values()))
        season_ids = list(season_centroids.keys())
        
        for idx, (season, centroid) in enumerate(season_centroids.items()):
            season_mask = df['id_season'] == season
            # Calcular similitud con otros centroides
            other_centroids = np.delete(centroid_matrix, idx, axis=0)
            similarities = cosine_similarity(centroid.reshape(1, -1), other_centroids).flatten()
            # Promedio de las 3 más similares
            top_similarities = np.sort(similarities)[-3:] if len(similarities) >= 3 else similarities
            df.loc[season_mask, 'similarity_to_nearest_seasons'] = top_similarities.mean()
    
    # Eliminar columna temporal
    df = df.drop('embedding_parsed', axis=1)
    
    print("Features de embeddings creadas exitosamente!")
    return df

# --- PASO 2: Definir las listas de features ---
# Primero creamos las features de embeddings avanzadas
print("Creando features avanzadas de embeddings...")
X = create_embedding_features(X_train_features)
X_test = create_embedding_features(X_test_features)

# (Ajusta esto según las columnas que realmente uses en tu 'X')
CATEGORICAL_FEATURES = [
    'id_season', 'aggregated_family', 'family', 'category', 'fabric', 
    'color_name', 'length_type', 'silhouette_type', 'waist_type', 
    'neck_lapel_type', 'sleeve_length_type', 'heel_shape_type', 
    'toecap_type', 'woven_structure', 'knit_structure', 'print_type', 
    'archetype', 'moment', 'embedding_cluster'  # Añadimos el cluster de embedding
]

NUMERICAL_FEATURES = [
    'life_cycle_length', 'num_stores', 'num_sizes', 'has_plus_sizes', 'price',
    'similarity_to_season_center',  # Nueva feature
    'avg_distance_within_season',   # Nueva feature
    'similarity_to_nearest_seasons'  # Nueva feature
]

# La columna de embedding se maneja por separado
EMBEDDING_COLUMN = ['image_embedding'] # Debe ser una lista para ColumnTransformer


# --- PASO 3: Crear los pipelines de preprocesamiento ---

# Pipeline para datos numéricos
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Rellenar faltantes con la mediana
    ('scaler', StandardScaler())                   # Escalar datos
])

# Pipeline para datos categóricos
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Rellenar faltantes con la moda
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # One-Hot Encoding
])

# Pipeline para los Embeddings:
# 1. Llama a nuestra función 'parse_embeddings'
# 2. Escala los valores (importante para PCA)
# 3. Aplica PCA para reducir de N dimensiones (ej. 256) a 50
embedding_pipeline = Pipeline(steps=[
    ('parser', FunctionTransformer(parse_embeddings)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=42)) # Aumentamos a 50 componentes
])

# --- PASO 4: Unir todo con ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ('embed', embedding_pipeline, EMBEDDING_COLUMN)
    ],
    remainder='drop' # Ignora columnas que no estén en las listas
)

# --- PASO 5: Definir el Modelo (LGBM Quantile) ---
lgbm_model = lgb.LGBMRegressor(
    objective='quantile',  # <-- Objetivo: Regresión de Cuantiles
    alpha=0.70,            # <-- Predecimos el percentil 70 (ajustado de 75)
    n_estimators=1500,     # Aumentamos árboles
    learning_rate=0.03,    # Reducimos learning rate
    num_leaves=45,         # Aumentamos complejidad
    max_depth=8,           # Limitamos profundidad
    min_child_samples=20,  # Mínimo de muestras por hoja
    subsample=0.8,         # Bagging
    colsample_bytree=0.8,  # Feature sampling
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=0.1,        # L2 regularization
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# --- PASO 6: Crear el Pipeline Final ---
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgbm_model)
])

print("¡Nuevo 'full_pipeline' (con Embeddings + Quantile) creado!")
print("---")


# --- PASO 7: Predicción Final y Archivo de Envío ---
# (Este es tu código original, que ahora usa el 'full_pipeline' de arriba)

print("Iniciando Paso 7: Predicción Final...")

# --- Tarea 1: Entrenar con TODOS los datos ---
print("Re-entrenando el modelo con el 100% de los datos de entrenamiento...")
# Asumiendo que 'X', 'y', 'X_test_features' y 'test_ids_for_submission'
# fueron cargados en pasos anteriores (1-6)
full_pipeline.fit(X, y)

print("¡Modelo final entrenado!")

# --- Tarea 2: Predecir sobre test_df_clean ---
print("Generando predicciones sobre los datos de test...")
# Usamos X_test que ya tiene las features de embeddings creadas
final_predictions = full_pipeline.predict(X_test)

print("¡Predicciones generadas!")

# --- Tarea 3: Post-Procesamiento (¡MUY IMPORTANTE!) ---
# La demanda no puede ser negativa.
final_predictions[final_predictions < 0] = 0
print("Predicciones negativas ajustadas a 0.")

# --- Tarea 4: Crear Archivo de Envío ---
# Usamos los 'test_ids_for_submission' que guardamos en el Paso 2
# y nuestras 'final_predictions'
submission_df = pd.DataFrame({
    'ID': test_ids_for_submission,
    'demand': final_predictions
})

# --- Tarea 5: Guardar el Archivo ---
submission_filename = 'submission_v3_season_embeddings.csv'
submission_df.to_csv(submission_filename, index=False, sep=',')

print("\n")
print("="*50)
print(f"¡Archivo '{submission_filename}' creado con éxito!")
print("="*50)
print("Vistazo al archivo de envío:")
print(submission_df.head())
print("\n")
print("¡Paso 7 completado!")
print(f"\nMejoras implementadas:")
print("- Clustering de embeddings por temporada")
print("- Similitud con centroide de temporada")
print("- Distancias intra-temporada")
print("- Similitud con temporadas cercanas")
print("- Modelo LGBM optimizado con regularización")
