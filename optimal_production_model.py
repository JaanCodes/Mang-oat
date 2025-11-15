"""
DATATHON 2025 - Optimal Production Model
=========================================
Modelo optimizado para predicción de demanda con penalización asimétrica.

Estrategia:
- Penaliza MÁS quedarse bajo del stock (lost sales)
- Mejor sobreestimar que subestimar
- Usa Gradient Boosting con quantile loss (alpha=0.6) para sesgo hacia arriba
- Validación temporal: entrenar en seasons 86,87,88 → validar en 89
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATATHON 2025 - OPTIMAL PRODUCTION MODEL")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1/7] Cargando datos...")
train = pd.read_csv('train.csv', delimiter=';')
test = pd.read_csv('test.csv', delimiter=';')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"  Train: {train.shape}")
print(f"  Test: {test.shape}")
print(f"  Target: {sample_submission.columns[1]}")  # 'Production' en el sample

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Feature Engineering...")

def extract_features(df, is_train=True):
    """
    Extrae features críticas para predicción de demanda.
    """
    df = df.copy()
    
    # --- Expandir image_embedding (512 dimensiones) ---
    if 'image_embedding' in df.columns:
        embedding_str = df['image_embedding'].fillna('')
        embeddings = []
        for emb in embedding_str:
            if emb:
                try:
                    embeddings.append([float(x) for x in emb.split(',')])
                except:
                    embeddings.append([0.0] * 512)
            else:
                embeddings.append([0.0] * 512)
        
        embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(512)])
        df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
        df.drop('image_embedding', axis=1, inplace=True)
    
    # --- Temporal features ---
    if 'phase_in' in df.columns:
        df['phase_in'] = pd.to_datetime(df['phase_in'], format='%d/%m/%Y', errors='coerce')
        df['phase_in_month'] = df['phase_in'].dt.month
        df['phase_in_quarter'] = df['phase_in'].dt.quarter
        df['phase_in_week'] = df['phase_in'].dt.isocalendar().week
    
    if 'phase_out' in df.columns:
        df['phase_out'] = pd.to_datetime(df['phase_out'], format='%d/%m/%Y', errors='coerce')
        df['phase_out_month'] = df['phase_out'].dt.month
    
    # --- Color features (RGB) ---
    if 'color_rgb' in df.columns:
        rgb_split = df['color_rgb'].str.split(',', expand=True)
        if rgb_split.shape[1] >= 3:
            df['color_r'] = pd.to_numeric(rgb_split[0], errors='coerce').fillna(128)
            df['color_g'] = pd.to_numeric(rgb_split[1], errors='coerce').fillna(128)
            df['color_b'] = pd.to_numeric(rgb_split[2], errors='coerce').fillna(128)
            
            # Brightness y saturation
            df['color_brightness'] = (df['color_r'] + df['color_g'] + df['color_b']) / 3
            df['color_saturation'] = df[['color_r', 'color_g', 'color_b']].max(axis=1) - df[['color_r', 'color_g', 'color_b']].min(axis=1)
        df.drop('color_rgb', axis=1, inplace=True)
    
    # --- Interaction features (CRÍTICOS para predicción de capacidad) ---
    if 'num_stores' in df.columns and 'num_sizes' in df.columns:
        df['total_capacity'] = df['num_stores'] * df['num_sizes']
        df['store_size_interaction'] = df['num_stores'] * df['num_sizes']
    
    if 'price' in df.columns and 'life_cycle_length' in df.columns:
        df['price_per_week'] = df['price'] / (df['life_cycle_length'] + 1)
        df['total_revenue_potential'] = df['price'] * df.get('total_capacity', 0)
    
    # --- Weekly demand/sales features (SOLO en train) ---
    if is_train:
        if 'weekly_demand' in df.columns and 'weekly_sales' in df.columns:
            df['demand_to_sales_ratio'] = df['weekly_demand'] / (df['weekly_sales'] + 1e-6)
            df['demand_to_sales_ratio'] = df['demand_to_sales_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Agregaciones por producto (solo si existe ID)
            if 'ID' in df.columns:
                df['cumulative_demand'] = df.groupby('ID')['weekly_demand'].transform('sum')
                df['cumulative_sales'] = df.groupby('ID')['weekly_sales'].transform('sum')
                df['avg_weekly_demand'] = df.groupby('ID')['weekly_demand'].transform('mean')
                df['max_weekly_demand'] = df.groupby('ID')['weekly_demand'].transform('max')
            else:
                df['cumulative_demand'] = df['weekly_demand']
                df['cumulative_sales'] = df['weekly_sales']
                df['avg_weekly_demand'] = df['weekly_demand']
                df['max_weekly_demand'] = df['weekly_demand']
    
    # Eliminar columnas no necesarias
    drop_cols = ['phase_in', 'phase_out', 'color_name']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    
    return df

train_fe = extract_features(train, is_train=True)
test_fe = extract_features(test, is_train=False)

print(f"  Features extraídas: {train_fe.shape[1]} columnas")

# ============================================================================
# 3. LABEL ENCODING PARA CATEGORICAS
# ============================================================================
print("\n[3/7] Encoding variables categóricas...")

categorical_cols = [
    'aggregated_family', 'family', 'category', 'fabric', 
    'length_type', 'silhouette_type', 'waist_type', 'neck_lapel_type', 
    'sleeve_length_type', 'heel_shape_type', 'toecap_type', 
    'woven_structure', 'knit_structure', 'print_type', 'archetype', 'moment'
]

label_encoders = {}
for col in categorical_cols:
    if col in train_fe.columns:
        le = LabelEncoder()
        # Fit en train + test combinados para evitar categorías desconocidas
        all_values = pd.concat([train_fe[col].fillna('Unknown'), test_fe[col].fillna('Unknown')])
        le.fit(all_values)
        
        train_fe[col] = le.transform(train_fe[col].fillna('Unknown'))
        test_fe[col] = le.transform(test_fe[col].fillna('Unknown'))
        label_encoders[col] = le

print(f"  {len(label_encoders)} variables codificadas")

# ============================================================================
# 4. PREPARAR TRAIN/VALIDATION SPLIT (TEMPORAL)
# ============================================================================
print("\n[4/7] Preparando Train/Validation split temporal...")

# Target = weekly_demand (demanda semanal real)
if 'weekly_demand' not in train_fe.columns:
    raise ValueError("ERROR: 'weekly_demand' no encontrada en train. Verificar datos.")

# Separación temporal: Seasons 86,87,88 → train | Season 89 → validation
train_seasons = train_fe[train_fe['id_season'] < 89]
val_seasons = train_fe[train_fe['id_season'] == 89]

print(f"  Train: {train_seasons.shape[0]} muestras (seasons < 89)")
print(f"  Validation: {val_seasons.shape[0]} muestras (season 89)")

# Separar features y target
feature_cols = [col for col in train_fe.columns if col not in ['ID', 'id_season', 'weekly_demand', 'weekly_sales', 'production']]

X_train = train_seasons[feature_cols].fillna(0)
y_train = train_seasons['weekly_demand']

X_val = val_seasons[feature_cols].fillna(0)
y_val = val_seasons['weekly_demand']

print(f"  Features: {len(feature_cols)}")
print(f"  Target distribution (train): min={y_train.min():.2f}, mean={y_train.mean():.2f}, max={y_train.max():.2f}")

# ============================================================================
# 5. ENTRENAR MODELO GRADIENT BOOSTING con QUANTILE LOSS
# ============================================================================
print("\n[5/7] Entrenando Gradient Boosting Regressor (Quantile Loss, alpha=0.6)...")
print("  → Penaliza MÁS subestimaciones (lost sales)")
print("  → Sesgo hacia SOBREESTIMACIÓN de demanda")

# Modelo con quantile loss (alpha=0.6 → 60% superior → sesgo hacia arriba)
gbr_model = GradientBoostingRegressor(
    loss='quantile',           # Quantile loss
    alpha=0.60,                 # 60th percentile (sesgo hacia arriba)
    n_estimators=300,          # Más árboles para capturar patrones
    max_depth=8,               # Profundidad controlada
    min_samples_split=50,      # Evitar overfitting
    min_samples_leaf=20,
    learning_rate=0.05,        # Learning rate conservador
    subsample=0.8,             # Stochastic boosting
    random_state=42,
    verbose=0
)

print("\n  Entrenando modelo...")
gbr_model.fit(X_train, y_train)

# Validación
y_train_pred = gbr_model.predict(X_train)
y_val_pred = gbr_model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n  TRAIN:")
print(f"    RMSE: {train_rmse:.4f}")
print(f"    MAE:  {train_mae:.4f}")
print(f"    R²:   {train_r2:.4f}")

print(f"\n  VALIDATION:")
print(f"    RMSE: {val_rmse:.4f}")
print(f"    MAE:  {val_mae:.4f}")
print(f"    R²:   {val_r2:.4f}")

# Análisis de sesgo (verificar que sobreestima)
overestimation_rate = (y_val_pred > y_val).mean()
avg_overestimation = (y_val_pred - y_val).mean()
print(f"\n  SESGO:")
print(f"    Overestimation Rate: {overestimation_rate*100:.1f}%")
print(f"    Avg Overestimation:  {avg_overestimation:.4f}")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n[6/7] Top 20 Features Importantes:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gbr_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.6f}")

# ============================================================================
# 7. PREDICCIÓN EN TEST Y GENERACIÓN DE SUBMISSION
# ============================================================================
print("\n[7/7] Prediciendo en test y generando submission...")

# Asegurar que test tenga las mismas columnas
X_test = test_fe[[col for col in feature_cols if col in test_fe.columns]].fillna(0)

# Predicción
test_predictions = gbr_model.predict(X_test)

# Clip negativo (demand no puede ser negativa)
test_predictions = np.clip(test_predictions, 0, None)

print(f"  Predicciones: min={test_predictions.min():.2f}, mean={test_predictions.mean():.2f}, max={test_predictions.max():.2f}")

# Crear submission en formato requerido
submission = pd.DataFrame({
    'ID': test['ID'],
    'Production': test_predictions  # Nota: sample_submission usa 'Production', no 'demand'
})

output_file = 'submission_optimal_gbr.csv'
submission.to_csv(output_file, index=False)

print(f"\n✓ Submission guardado: {output_file}")
print(f"  Shape: {submission.shape}")
print(f"  Primeras 5 filas:")
print(submission.head())

print("\n" + "="*80)
print("COMPLETADO - Modelo optimizado para MINIMIZAR LOST SALES")
print("Estrategia: Quantile Loss (60th percentile) → Sesgo hacia sobreestimación")
print("="*80)
