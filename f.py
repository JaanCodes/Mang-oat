# --- Datathon Mango: Predicción de Demanda (V6) ---
#
# ESTRATEGIA:
# 1. Agregación: De datos semanales a datos por producto.
# 2. Ingeniería de Features: Creación de "features de contexto" (promedios históricos).
# 3. Transformación del Target: Uso de np.log1p() para normalizar la demanda.
# 4. Modelo: LGBMRegressor.

print("Iniciando script de modelo V6...")

# --- Paso 0: Importar Librerías ---
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Modelo y Métricas
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ignorar warnings para una salida más limpia
warnings.filterwarnings('ignore')

def main():
    """Función principal para ejecutar todo el pipeline."""
    
    # --- Paso 1: Cargar Datos ---
    print("\n--- Paso 1: Cargando Datos ---")
    data_dir = Path('.')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    
    try:
        train_df = pd.read_csv(train_path, sep=';')
        test_df = pd.read_csv(test_path, sep=';')
    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos en {data_dir.resolve()}")
        print("Asegúrate de que 'train.csv' y 'test.csv' están en la misma carpeta que el script.")
        return

    print("Datos cargados con éxito.")
    print(f"Train:   {train_df.shape}")
    print(f"Test:    {test_df.shape}")

    # --- Paso 2: Agregación de Datos (El Paso Crítico) ---
    print("\n--- Paso 2: Agregando Datos Semanales a Nivel de Producto ---")
    
    # 1. Crear el Target (y_train_agg)
    y_train_agg = train_df.groupby('ID')['weekly_demand'].sum().to_frame(name='total_demand')

    # 2. Crear las Features (X_train_agg_full)
    X_train_agg_full = train_df.groupby('ID').first()

    # 3. Preparar el set de Test (X_test_full)
    X_test_full = test_df.set_index('ID') 

    print("Agregación completada.")

    # --- Paso 3: Ingeniería de Características (Features) ---
    print("\n--- Paso 3: Creando Nuevas Características ---")
    
    # --- Parte A: Features de Fecha ---
    X_features = X_train_agg_full.copy()
    X_test_features = X_test_full.copy()

    for df in [X_features, X_test_features]:
        df['phase_in'] = pd.to_datetime(df['phase_in'], errors='coerce')
        df['start_month'] = df['phase_in'].dt.month

    median_month = X_features['start_month'].median()
    X_features['start_month'] = X_features['start_month'].fillna(median_month)
    X_test_features['start_month'] = X_test_features['start_month'].fillna(median_month)
    print("Feature 'start_month' creada.")

    # --- Parte B: Features de Agregación (Contexto) ---
    train_data_master = X_train_agg_full.join(y_train_agg)

    avg_demand_by_family = train_data_master.groupby('family')['total_demand'].mean().to_dict()
    avg_demand_by_category = train_data_master.groupby('category')['total_demand'].mean().to_dict()
    avg_prod_by_family = train_data_master.groupby('family')['Production'].mean().to_dict()
    avg_price_by_family = train_data_master.groupby('family')['price'].mean().to_dict()

    global_avg_demand = y_train_agg['total_demand'].mean()
    global_avg_prod = train_data_master['Production'].mean()
    global_avg_price = train_data_master['price'].mean()

    for df in [X_features, X_test_features]:
        df['family_avg_demand'] = df['family'].map(avg_demand_by_family).fillna(global_avg_demand)
        df['category_avg_demand'] = df['category'].map(avg_demand_by_category).fillna(global_avg_demand)
        df['family_avg_prod'] = df['family'].map(avg_prod_by_family).fillna(global_avg_prod)
        df['family_avg_price'] = df['family'].map(avg_price_by_family).fillna(global_avg_price)
        df['price_vs_family'] = df['price'] - df['family_avg_price']
    
    print("Features de agregación (promedios históricos) creadas.")

    # --- Paso 4: Preprocesamiento y Pipeline ---
    print("\n--- Paso 4: Definiendo el Pipeline de Preprocesamiento ---")
    
    # --- 1. Definir las listas de features --- 
    numeric_features = [
        'num_stores', 'price', 'life_cycle_length', 'num_sizes', 'start_month'
    ]
    agg_features = [
        'family_avg_demand', 'category_avg_demand', 'family_avg_prod',
        'family_avg_price', 'price_vs_family'
    ]
    all_numeric_features = numeric_features + agg_features

    categorical_features = [
        'moment', 'archetype', 'neck_lapel_type', 'print_type', 'has_plus_sizes',
        'knit_structure', 'waist_type', 'silhouette_type', 'family', 'length_type',
        'color_name', 'category', 'woven_structure', 'id_season', 
        'aggregated_family', 'sleeve_length_type', 'fabric'
    ]

    features_to_use = all_numeric_features + categorical_features
    
    # --- 2. Crear los DataFrames finales para el modelo ---
    X_train_final = X_features[features_to_use]
    # Asegurarnos de que test tiene las mismas columnas (algunas pueden faltar, ej. 'Unnamed')
    X_test_final = X_test_features[[col for col in features_to_use if col in X_test_features.columns]]
    y_train_final = y_train_agg['total_demand']

    # --- 3. Construir los transformadores ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    print("Preprocesador creado.")

    # --- Paso 5: Entrenamiento y Validación del Modelo ---
    print("\n--- Paso 5: Entrenando y Validando el Modelo ---")
    
    # 1. Transformar el Target (y)
    y_train_log = np.log1p(y_train_final)

    # 2. Dividir los datos
    # CORRECCIÓN: Desempaquetamos las 6 variables que devuelve train_test_split
    # (aunque solo usaremos 4 de ellas)
    X_train_local, X_val_local, y_train_local_log, y_val_local_log, y_train_local_real, y_val_local_real = train_test_split(
        X_train_final, 
        y_train_log, 
        y_train_final, # y original para validación
        test_size=0.2, 
        random_state=42
    )

    # 3. Definir el Modelo LGBM
    model_lgbm = LGBMRegressor(
        objective='regression_l1', # Optimiza para MAE
        n_estimators=500,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )

    # 4. Crear el Pipeline Final
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_lgbm)
    ])

    # 5. Entrenar el Modelo
    print("Entrenando el modelo V6 (LGBM + Log + Agg-Features)...")
    full_pipeline.fit(X_train_local, y_train_local_log)
    print("¡Modelo V6 entrenado!")

    # 6. Evaluar Localmente
    y_pred_log = full_pipeline.predict(X_val_local)
    y_pred_real = np.expm1(y_pred_log) # Invertir el log
    mae_v6 = mean_absolute_error(y_val_local_real, y_pred_real)
    
    print("-" * 30)
    print(f"VALIDACIÓN LOCAL - MAE: {mae_v6:.4f}")
    print("-" * 30)

    # --- Paso 6: Generación del Archivo de Envío Final ---
    print("\n--- Paso 6: Generando Archivo de Envío Final ---")
    
    # 1. Re-entrenar con TODOS los datos
    print("Re-entrenando el modelo V6 con el 100% de los datos...")
    full_pipeline.fit(X_train_final, y_train_log)
    print("¡Modelo final entrenado!")

    # 2. Predecir sobre el set de Test
    final_predictions_log = full_pipeline.predict(X_test_final)

    # 3. Invertir la transformación Logarítmica
    final_predictions_real = np.expm1(final_predictions_log)

    # 4. Post-Procesamiento (Limpieza)
    final_predictions_real[final_predictions_real < 0] = 0
    print("Predicciones finales generadas y limpiadas (negativos -> 0).")

    # 5. Crear el DataFrame de Envío
    submission_df = pd.DataFrame({
        'ID': X_test_final.index,
        'demand': final_predictions_real
    })

    # 6. Guardar el archivo
    submission_filename = 'submission_final_v6.csv'
    submission_df.to_csv(submission_filename, index=False, sep=',')

    print("\n" + "="*50)
    print(f"¡PROCESO COMPLETADO! Archivo '{submission_filename}' creado con éxito.")
    print("="*50)
    print("Vistazo al archivo de envío:")
    print(submission_df.head())

if __name__ == "__main__":
    main()