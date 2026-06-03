"""
DETECTOR DE HIPOGLUCEMIA - MODELO DE CLASIFICACIÓN BINARIA
===========================================================

ENFOQUE COMPLETAMENTE NUEVO: Clasificación en vez de Regresión

Problema: ¿Habrá hipoglucemia (<70 mg/dL) en los próximos 30 minutos?
Target: Binario (0 = No, 1 = Sí)

Ventajas sobre regresión:
1. Optimiza directamente para detectar hipoglucemias (no MAE global)
2. Métricas apropiadas: Recall, Precision, F1-score
3. Class weights para manejar desbalance
4. Threshold ajustable para balance sensibilidad/especificidad

Autor: Análisis de Sesgos LSTM
Fecha: 2026-04-11
"""

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / 'models'
DEFAULT_DATA_DIRS = [
    REPO_ROOT,
    REPO_ROOT / 'data' / 'raw' / 'demo',
]
MODEL_PATH = MODELS_DIR / 'lstm_hypoglycemia_classifier.h5'
THRESHOLD_PATH = MODELS_DIR / 'optimal_threshold.npy'
DEFAULT_THRESHOLD = 0.5

# ==========================================
# 1. FUNCIONES DE PREPARACIÓN DE DATOS
# ==========================================

def _candidate_data_roots(base_path=None):
    """Devuelve raíces de datos candidatas para el formato original y el demo del repo."""
    roots = []
    if base_path is not None:
        roots.append(Path(base_path).expanduser().resolve())
    roots.extend(DEFAULT_DATA_DIRS)

    deduped = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _find_patient_ids(data_roots):
    """Encuentra IDs de paciente en carpetas originales o en `data/raw/demo`."""
    id_pattern = re.compile(r'UoMGlucose(\d+)\.csv$')
    patient_ids = set()

    for root in data_roots:
        if not root.exists():
            continue
        for glucose_file in root.rglob('UoMGlucose*.csv'):
            match = id_pattern.search(glucose_file.name)
            if match:
                patient_ids.add(int(match.group(1)))

    return sorted(patient_ids)


def _find_first_existing(data_roots, patient_id, file_name, legacy_relative_path):
    """Busca un archivo por nombre plano o por la ruta histórica del dataset."""
    candidates = []
    for root in data_roots:
        candidates.extend([
            root / file_name,
            root / legacy_relative_path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for root in data_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(file_name))
        if matches:
            return matches[0]

    raise FileNotFoundError(f'No se encontró {file_name} para el paciente {patient_id}')


def _read_csv(path):
    """Lee CSVs exportados con o sin BOM en cabecera."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.replace('\ufeff', '') for c in df.columns]
    return df


def process_full_data(base_path=None):
    """
    Procesa datos de pacientes desde el layout original o desde `data/raw/demo`.

    El script original asumía carpetas hermanas como `Glucose Data/`.
    Esta versión también detecta los CSV planos subidos al repo, por ejemplo
    `data/raw/demo/UoMGlucose2302.csv`.
    """
    data_roots = _candidate_data_roots(base_path)
    patient_ids = _find_patient_ids(data_roots)
    all_patients_data = []

    if not patient_ids:
        roots = ', '.join(str(root) for root in data_roots)
        raise FileNotFoundError(f'No se encontraron archivos UoMGlucose*.csv en: {roots}')

    for p_id in patient_ids:
        try:
            glucose_path = _find_first_existing(
                data_roots,
                p_id,
                f'UoMGlucose{p_id}.csv',
                f'Glucose Data/UoMGlucose{p_id}.csv',
            )
            bolus_path = _find_first_existing(
                data_roots,
                p_id,
                f'UoMBolus{p_id}.csv',
                f'Insulin Data/Bolus Data/UoMBolus{p_id}.csv',
            )
            meals_path = _find_first_existing(
                data_roots,
                p_id,
                f'UoMNutrition{p_id}.csv',
                f'Nutrition Data/UoMNutrition{p_id}.csv',
            )
            activity_path = _find_first_existing(
                data_roots,
                p_id,
                f'UoMActivity{p_id}.csv',
                f'Activity Data/UoMActivity{p_id}.csv',
            )

            # Carga de archivos
            df_bg = _read_csv(glucose_path)
            df_bolus = _read_csv(bolus_path)
            df_meals = _read_csv(meals_path)
            df_activity = _read_csv(activity_path)

            def to_5min_grid(df, col):
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce').dt.tz_localize(None)
                df = df.dropna(subset=[col]).copy()
                df['ts_grid'] = df[col].dt.floor('5min')
                return df

            df_bg = to_5min_grid(df_bg, 'bg_ts')
            df_bolus = to_5min_grid(df_bolus, 'bolus_ts')
            df_meals = to_5min_grid(df_meals, 'meal_ts')
            df_activity = to_5min_grid(df_activity, 'activity_ts')

            master_index = pd.date_range(
                start=df_bg['ts_grid'].min(),
                end=df_bg['ts_grid'].max(),
                freq='5min',
            )
            df_p = pd.DataFrame(index=master_index)

            # Join de datos fisiológicos
            df_bg_clean = df_bg.drop_duplicates('ts_grid').set_index('ts_grid')
            df_p['glucose'] = df_bg_clean['value'].reindex(df_p.index).interpolate(method='time') * 18
            df_p['bolus'] = df_bolus.groupby('ts_grid')['bolus_dose'].sum().reindex(df_p.index).fillna(0)
            df_p = df_p.join(df_meals.groupby('ts_grid')[['carbs_g']].sum()).fillna(0)
            df_p = df_p.join(df_activity.groupby('ts_grid')[['step_count']].sum()).fillna(0)

            # Features temporales e IOB simple
            df_p['iob'] = df_p['bolus'].rolling(window=48, min_periods=1).sum()
            df_p['p_id'] = p_id

            # Target BINARIO: ¿Habrá hipoglucemia en 30 min?
            df_p['glucose_future'] = df_p['glucose'].shift(-6)  # 30 min adelante
            df_p['target_hypo'] = (df_p['glucose_future'] < 70).astype(int)  # 1 = hipoglucemia, 0 = no

            all_patients_data.append(df_p.dropna())
            print(f'Paciente {p_id}: OK ({glucose_path.parent})')

        except Exception as e:
            print(f'Error en Paciente {p_id}: {e}')
            continue

    if not all_patients_data:
        raise RuntimeError('Se encontraron pacientes, pero ninguno pudo procesarse correctamente.')

    return pd.concat(all_patients_data)


def create_sequences_for_classification(df, lookback=48):
    """
    Crea secuencias para clasificación binaria
    
    A diferencia de regresión, aquí NO hacemos muestreo diferencial.
    Queremos todas las muestras para que el modelo aprenda bien ambas clases.
    """
    X_cols = [c for c in df.columns if c not in ['glucose_future', 'target_hypo', 'p_id']]
    X_data = df[X_cols].values
    y_data = df['target_hypo'].values
    
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(df)):
        X_seq.append(X_data[i-lookback:i])
        y_seq.append(y_data[i])
    
    return np.array(X_seq), np.array(y_seq)


# ==========================================
# 2. MODELO LSTM PARA CLASIFICACIÓN
# ==========================================

def build_hypoglycemia_classifier(lookback, n_features):
    """
    LSTM para clasificación binaria de hipoglucemia
    
    Arquitectura optimizada para detección de eventos raros
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), 
                     input_shape=(lookback, n_features)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # ← Sigmoid para clasificación binaria
    ])
    
    return model


# ==========================================
# 3. MÉTRICAS Y EVALUACIÓN
# ==========================================

def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluación completa del clasificador con múltiples métricas
    """
    # Predicciones (probabilidades)
    y_pred_proba = model.predict(X_test).flatten()
    
    # Predicciones binarias con threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("\n" + "="*70)
    print("   EVALUACIÓN DEL CLASIFICADOR DE HIPOGLUCEMIA")
    print("="*70)
    print(f"   Threshold usado: {threshold:.2f}")
    print("="*70)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métricas clínicas
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n   📊 MÉTRICAS CLÍNICAS:")
    print(f"   Sensibilidad (Recall):    {sensitivity:.3f}  ← % de hipos detectadas")
    print(f"   Especificidad:            {specificity:.3f}  ← % de no-hipos correctas")
    print(f"   Valor Predictivo Positivo: {ppv:.3f}  ← Precisión de alertas")
    print(f"   Valor Predictivo Negativo: {npv:.3f}  ← Confianza en 'seguro'")
    print(f"   F1-Score:                 {f1:.3f}  ← Balance precision/recall")
    
    # AUC-ROC
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   AUC-ROC:                  {auc:.3f}  ← Capacidad discriminativa")
    except:
        print(f"   AUC-ROC:                  N/A")
    
    print(f"\n   🎯 MATRIZ DE CONFUSIÓN:")
    print(f"                    Predicho No    Predicho Sí")
    print(f"   Real No (TN/FP):    {tn:5d}         {fp:5d}")
    print(f"   Real Sí (FN/TP):    {fn:5d}         {tp:5d}")
    
    print(f"\n   📈 INTERPRETACIÓN:")
    print(f"   Verdaderos Positivos (TP): {tp} hipoglucemias detectadas correctamente")
    print(f"   Falsos Negativos (FN):     {fn} hipoglucemias NO detectadas ⚠️")
    print(f"   Falsos Positivos (FP):     {fp} falsas alarmas")
    print(f"   Verdaderos Negativos (TN): {tn} no-hipos correctas")
    
    print("="*70)
    
    # Distribución de clases
    n_hypo = y_test.sum()
    n_total = len(y_test)
    print(f"\n   📊 DISTRIBUCIÓN EN TEST:")
    print(f"   Hipoglucemias: {n_hypo} ({100*n_hypo/n_total:.1f}%)")
    print(f"   No-hipoglucemias: {n_total-n_hypo} ({100*(n_total-n_hypo)/n_total:.1f}%)")
    print("="*70)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'cm': cm
    }


def find_optimal_threshold(model, X_val, y_val, beta=2.0):
    """
    Encuentra el threshold óptimo que maximiza F-beta score
    
    beta=2.0 → F2-Score: Prioriza recall 2x sobre precision
    Apropiado para detección de hipoglucemia donde:
    - Falso Negativo (no detectar hipo) = MUY GRAVE
    - Falso Positivo (falsa alarma) = Molestia tolerable
    """
    from sklearn.metrics import fbeta_score
    
    y_pred_proba = model.predict(X_val).flatten()
    
    # Probar diferentes thresholds
    thresholds = np.arange(0.05, 0.95, 0.05)
    f_scores = []
    recalls = []
    precisions = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f_beta = fbeta_score(y_val, y_pred, beta=beta)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred) if y_pred.sum() > 0 else 0
        
        f_scores.append(f_beta)
        recalls.append(recall)
        precisions.append(precision)
    
    optimal_idx = np.argmax(f_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f_beta = f_scores[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    
    print(f"\n🎯 Threshold Óptimo: {optimal_threshold:.2f}")
    print(f"   F{beta}-Score: {optimal_f_beta:.3f}")
    print(f"   Recall:    {optimal_recall:.3f}")
    print(f"   Precision: {optimal_precision:.3f}")
    
    # Mostrar trade-off
    print(f"\n📊 Trade-off Analysis (top 5 thresholds):")
    sorted_indices = np.argsort(f_scores)[::-1][:5]
    for idx in sorted_indices:
        print(f"   Threshold {thresholds[idx]:.2f}: F{beta}={f_scores[idx]:.3f}, "
              f"Recall={recalls[idx]:.3f}, Precision={precisions[idx]:.3f}")
    
    return optimal_threshold


def generate_classification_gif(model, X_seq, y_true, scaler, glucose_col_idx=0, threshold=0.5, num_frames=200):
    """
    Genera un GIF animado de la predicción paso a paso para clasificación.
    Muestra la glucosa real recuperada y la probabilidad de hipoglucemia predicha a futuro.
    """
    import matplotlib.animation as animation
    
    print("\n🎬 Generando animación GIF de alta fidelidad...")
    X_plot = X_seq[:num_frames]
    
    # Predecir probabilidades
    y_prob = model.predict(X_plot, verbose=0).flatten()
    
    # Desescalar glucosa (el último valor de cada secuencia de lookback)
    glucose_scaled = np.array([x[-1, glucose_col_idx] for x in X_plot])
    glucose_real = glucose_scaled * scaler.scale_[glucose_col_idx] + scaler.mean_[glucose_col_idx]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_ylim(40, 250)
    ax1.set_facecolor('#f8f9fa')
    ax1.set_ylabel("Glucosa Real (mg/dL)", fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("Riesgo Predicho en 30m (%)", color='red', fontweight='bold')
    
    # Elementos a dibujar
    line_gluc, = ax1.plot([], [], 'k-', lw=2, label='Glucosa (Pasado)', alpha=0.7)
    point_now, = ax1.plot([], [], 'ko', markersize=8)
    line_risk, = ax2.plot([], [], 'r--', lw=2, label='Riesgo de Hipo (IA)', alpha=0.8)
    line_proj, = ax2.plot([], [], 'r:', lw=1, alpha=0.5) 
    
    ax1.axhline(70, color='royalblue', linestyle='-', alpha=0.8, label='Límite Hipo (70)')
    ax2.axhline(threshold * 100, color='red', linestyle=':', alpha=0.4, label='Umbral de Alerta Médica')
    
    txt_info = ax1.text(0.05, 0.82, '', transform=ax1.transAxes, fontsize=11, 
                       fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
                       
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5)
    
    def init():
        line_gluc.set_data([], [])
        line_risk.set_data([], [])
        line_proj.set_data([], [])
        point_now.set_data([], [])
        return line_gluc, line_risk, line_proj, point_now, txt_info
        
    def animate(i):
        if i < 2: return line_gluc, line_risk, line_proj, point_now, txt_info
        
        # Real
        x_past = np.arange(0, i + 1)
        y_past_gluc = glucose_real[:i + 1]
        
        # Histórico de Riesgos (dibujamos el riesgo *donde se predijo que ocurriría*, +6 pasos)
        x_risk = np.arange(6, i + 7)
        y_past_risk = y_prob[:i + 1] * 100
        
        # Línea de proyección actual (radar puntual para este segundo)
        val_risk = y_prob[i] * 100
        x_proj = [i, i + 6]
        # mapeamos la glucosa actual al mismo eje 2 solo para visualizar el salto (opcional, lo dejamos entre 0 y el riesgo)
        y_proj = [0, val_risk]  # Desde abajo apuntando al riesgo futuro
        
        line_gluc.set_data(x_past, y_past_gluc)
        line_risk.set_data(x_risk, y_past_risk)
        # line_proj.set_data(x_proj, y_proj) # Desactivado por limpieza, pero útil para radar
        point_now.set_data([i], [glucose_real[i]])
        
        val_gluc = glucose_real[i]
        status = "¡ALERTA HIPO!" if (val_risk >= threshold * 100) else "ESTABLE"
        
        txt_info.set_text(f"TIEMPO REAL\nGlucosa: {val_gluc:.0f} mg/dL\nRiesgo a 30m: {val_risk:.1f}%\nEstado: {status}")
        ax1.set_xlim(max(0, i - 40), i + 15)
        
        return line_gluc, line_risk, line_proj, point_now, txt_info
        
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True, interval=100)
    filename = ML_DIR / 'prediccion_clasificacion_dinamica.gif'
    ani.save(filename, writer='pillow', fps=5)
    plt.close()
    print(f"✅ ¡Éxito! GIF animado mejorado guardado como '{filename}'")


# ==========================================
# 4. SCRIPT PRINCIPAL
# ==========================================

def main():
    """
    Ejecuta el pipeline completo de clasificación de hipoglucemia
    """
    print("\n" + "="*70)
    print("   DETECTOR DE HIPOGLUCEMIA - CLASIFICACIÓN BINARIA")
    print("="*70)
    
    # 1. Cargar y preparar datos
    print("\n📁 Cargando datos...")
    df_all = process_full_data()
    
    # 2. Análisis de distribución de clases
    n_hypo_total = df_all['target_hypo'].sum()
    n_total = len(df_all)
    print(f"\n📊 Distribución de clases en dataset completo:")
    print(f"   Hipoglucemias: {n_hypo_total} ({100*n_hypo_total/n_total:.1f}%)")
    print(f"   No-hipoglucemias: {n_total-n_hypo_total} ({100*(n_total-n_hypo_total)/n_total:.1f}%)")
    
    # 3. Escalado
    print("\n⚙️  Escalando features...")
    scaler = StandardScaler()
    X_features = [c for c in df_all.columns if c not in ['glucose_future', 'target_hypo', 'p_id']]
    df_all[X_features] = scaler.fit_transform(df_all[X_features])
    
    # 4. Separar train/test por paciente cuando hay varios pacientes.
    # Si solo existe un paciente demo, se usa un split cronológico 80/20.
    print("\n✂️  Separando train/test...")
    p_ids = sorted(df_all['p_id'].unique())
    LOOKBACK = 48

    if len(p_ids) > 1:
        train_ids = p_ids[:int(len(p_ids) * 0.8)]
        test_ids = p_ids[int(len(p_ids) * 0.8):]

        X_train_list, y_train_list = [], []
        for pid in train_ids:
            X_p, y_p = create_sequences_for_classification(
                df_all[df_all['p_id'] == pid],
                lookback=LOOKBACK,
            )
            if len(X_p) > 0:
                X_train_list.append(X_p)
                y_train_list.append(y_p)

        X_test_list, y_test_list = [], []
        for pid in test_ids:
            X_p, y_p = create_sequences_for_classification(
                df_all[df_all['p_id'] == pid],
                lookback=LOOKBACK,
            )
            if len(X_p) > 0:
                X_test_list.append(X_p)
                y_test_list.append(y_p)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        split_idx = int(len(df_all) * 0.8)
        train_df = df_all.iloc[:split_idx]
        test_df = df_all.iloc[split_idx:]
        X_train, y_train = create_sequences_for_classification(train_df, lookback=LOOKBACK)
        X_test, y_test = create_sequences_for_classification(test_df, lookback=LOOKBACK)

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError('No hay suficientes muestras para crear secuencias de train/test.')
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test:  {len(X_test)} muestras")
    
    # Distribución en train
    n_hypo_train = y_train.sum()
    print(f"\n   Train - Hipoglucemias: {n_hypo_train} ({100*n_hypo_train/len(y_train):.1f}%)")
    print(f"   Test  - Hipoglucemias: {y_test.sum()} ({100*y_test.sum()/len(y_test):.1f}%)")
    
    # 5. Calcular class weights
    print("\n⚖️  Calculando class weights para balancear clases...")
    present_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=present_classes,
        y=y_train,
    )
    class_weights = {int(cls): weight for cls, weight in zip(present_classes, class_weights_array)}
    class_weights.setdefault(0, 1.0)
    class_weights.setdefault(1, 1.0)
    print(f"   Peso clase 0 (no-hipo): {class_weights[0]:.2f}")
    print(f"   Peso clase 1 (hipo):    {class_weights[1]:.2f}")
    
    # 6. Construir y entrenar modelo O cargar existente
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("\n" + "="*70)
        print("   📥 RECUPERANDO MODELO PREVIAMENTE ENTRENADO...")
        print("="*70)
        model = load_model(MODEL_PATH)
        if THRESHOLD_PATH.exists():
            optimal_threshold = float(np.load(THRESHOLD_PATH))
            print(f"   ✅ Threshold óptimo '{optimal_threshold:.2f}' restaurado!")
        else:
            optimal_threshold = DEFAULT_THRESHOLD
            print(f"   ⚠️  No se encontró '{THRESHOLD_PATH.name}'. Usando threshold por defecto: {DEFAULT_THRESHOLD:.2f}")
        print(f"   ✅ Modelo '{MODEL_PATH}' cargado correctamente!")

    else:
        print("\n🏗️  Construyendo clasificador LSTM...")
        model = build_hypoglycemia_classifier(LOOKBACK, len(X_features))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # ← Loss para clasificación binaria
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("\n🎓 Entrenando clasificador con class weights...")
        monitor = EarlyStopping(
            monitor='val_auc',  # Monitorear AUC en vez de loss
            patience=15,
            restore_best_weights=True,
            mode='max'  # Maximizar AUC
        )
        
        history = model.fit(
            X_train, y_train,
            class_weight=class_weights,  # ← Class weights para balancear
            epochs=10,
            batch_size=64,
            validation_split=0.1,
            callbacks=[monitor],
            verbose=1
        )
        
        # 7. Encontrar threshold óptimo en validation
        print("\n🔍 Buscando threshold óptimo con F-beta Score...")
        print("   Beta alto = Prioriza RECALL (minimiza falsos negativos)")
        val_size = int(len(X_train) * 0.1)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        
        BETA = 5.0
        print(f"   Usando F{BETA}-Score (prioriza recall {BETA}x sobre precision)")
        optimal_threshold = find_optimal_threshold(model, X_val, y_val, beta=BETA)
        
        # Guardar modelo para la proxima vez
        print("\n💾 Guardando clasificador...")
        model.save(MODEL_PATH)
        np.save(THRESHOLD_PATH, optimal_threshold)
        print(f"   ✅ Modelo guardado como '{MODEL_PATH}'")
    
    # 8. Evaluación en test con threshold óptimo
    print("\n📊 Evaluando en test set...")
    results = evaluate_classifier(model, X_test, y_test, threshold=optimal_threshold)
    
    # 9. Comparación con modelo de regresión
    print("\n" + "="*70)
    print("   COMPARACIÓN: REGRESIÓN vs CLASIFICACIÓN")
    print("="*70)
    print("   MODELO DE REGRESIÓN (V1-V3):")
    print("   Sensibilidad: 14.2% - 15.0% (detecta 18-19 de 127 hipos)")
    print("   Problema: Optimiza MAE global, ignora hipoglucemias")
    print("\n   MODELO DE CLASIFICACIÓN (V4):")
    print(f"   Sensibilidad: {results['sensitivity']*100:.1f}% (detecta {int(results['sensitivity']*y_test.sum())} de {y_test.sum()} hipos)")
    print(f"   F1-Score: {results['f1']:.3f}")
    print(f"   Problema: Optimiza directamente para detectar hipoglucemias ✅")
    print("="*70)
    
    # 10. Visualizar matriz de confusión
    print("\n📊 MATRIZ DE CONFUSIÓN VISUAL:")
    print("="*70)
    cm = results['cm']
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular porcentajes
    total = tn + fp + fn + tp
    tn_pct = 100 * tn / total
    fp_pct = 100 * fp / total
    fn_pct = 100 * fn / total
    tp_pct = 100 * tp / total
    
    print(f"\n                    PREDICCIÓN")
    print(f"                No Hipo      Sí Hipo")
    print(f"           ┌─────────────┬─────────────┐")
    print(f"  No Hipo  │  TN: {tn:5d}  │  FP: {fp:5d}  │")
    print(f"  REAL     │  {tn_pct:5.1f}%     │  {fp_pct:5.1f}%     │")
    print(f"           ├─────────────┼─────────────┤")
    print(f"  Sí Hipo  │  FN: {fn:5d}  │  TP: {tp:5d}  │")
    print(f"           │  {fn_pct:5.1f}%     │  {tp_pct:5.1f}%     │")
    print(f"           └─────────────┴─────────────┘")
    
    print(f"\n   ✅ Verdaderos Negativos (TN): {tn} - Correctamente identificó NO hipoglucemia")
    print(f"   ✅ Verdaderos Positivos (TP): {tp} - Correctamente identificó hipoglucemia")
    print(f"   ⚠️  Falsos Positivos (FP): {fp} - Falsa alarma (predijo hipo, pero no era)")
    print(f"   ❌ Falsos Negativos (FN): {fn} - PELIGROSO (no detectó hipoglucemia real)")
    
    print("\n   📊 TASAS:")
    print(f"   Tasa de Detección (Recall):     {100*tp/(tp+fn):.1f}% de hipos detectadas")
    print(f"   Tasa de Falsas Alarmas:         {100*fp/(fp+tn):.1f}% de no-hipos")
    print(f"   Precisión de Alertas (PPV):     {100*tp/(tp+fp):.1f}% de alertas son reales")
    
    # Análisis de Falsos Negativos
    print("\n   🚨 ANÁLISIS DE FALSOS NEGATIVOS (FN):")
    total_hypos = tp + fn
    fn_rate = 100 * fn / total_hypos
    print(f"   Total hipoglucemias reales: {total_hypos}")
    print(f"   Detectadas (TP): {tp} ({100*tp/total_hypos:.1f}%)")
    print(f"   NO detectadas (FN): {fn} ({fn_rate:.1f}%) ← OBJETIVO: MINIMIZAR")
    
    if fn_rate > 20:
        print(f"   ⚠️  ADVERTENCIA: {fn_rate:.1f}% de hipos no detectadas es alto")
        print(f"   💡 SUGERENCIA: Aumenta BETA a 5.0 o reduce threshold manualmente")
    elif fn_rate > 10:
        print(f"   ⚠️  Aceptable pero mejorable: {fn_rate:.1f}% de hipos no detectadas")
    else:
        print(f"   ✅ EXCELENTE: Solo {fn_rate:.1f}% de hipos no detectadas")
    
    print("="*70)
    
    # 11. Generar la gráfica animada GIF (Simulación visual)
    print("\n" + "="*70)
    print("   🎞️  MÓDULO DE SIMULACIÓN VISUAL (GIF)")
    print("="*70)
    
    # Buscamos un trozo del test estático que tenga riesgo clínico para que sea interesante
    hypo_indices = np.where(y_test == 1)[0]
    start_idx = max(0, hypo_indices[0] - 80) if len(hypo_indices) > 0 else 0
    
    X_plot = X_test[start_idx : start_idx + 200]
    y_plot_true = y_test[start_idx : start_idx + 200]
    
    glucose_col_idx = X_features.index('glucose')
    
    generate_classification_gif(
        model, X_plot, y_plot_true, 
        scaler=scaler, 
        glucose_col_idx=glucose_col_idx, 
        threshold=optimal_threshold, 
        num_frames=len(X_plot)
    )

    print("\n" + "="*70)
    print("   ✅ PIPELINE COMPLETADO CON ÉXITO")
    print("="*70)


if __name__ == "__main__":
    main()
