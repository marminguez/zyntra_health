"""
Generates synthetic training data calibrated against published
Ohio T1DM dataset statistics for use until real dataset access is granted.
Statistics source: Marling & Bunescu (2020), PMC7881904
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_PATIENTS = 12
DAYS = 56  # 8 weeks like Ohio T1DM
SAMPLES_PER_DAY = 288  # 5-min intervals

def simulate_patient(patient_id: int) -> pd.DataFrame:
    n = DAYS * SAMPLES_PER_DAY
    t = np.arange(n)

    # Glucose: mean ~140 mg/dL, std ~40, with circadian pattern
    circadian = 15 * np.sin(2 * np.pi * t / SAMPLES_PER_DAY - np.pi / 2)
    noise = np.random.normal(0, 1, n)
    smoothed_series = pd.Series(noise).ewm(alpha=0.04).mean()
    noise_smoothed = np.asarray(smoothed_series)
    noise_scaled = (noise_smoothed - np.mean(noise_smoothed)) / float(np.std(noise_smoothed)) * 25
    glucose = 140 + circadian + noise_scaled
    glucose = np.clip(glucose, 40, 400)

    # HRV rmssd: mean ~35ms nocturnal, std ~12
    hrv = np.random.normal(35, 12, n)
    hrv = np.clip(hrv, 8, 100)

    # Sleep score: 0-100, sampled once per day
    sleep_score = np.repeat(np.random.normal(68, 14, DAYS), SAMPLES_PER_DAY)
    sleep_score = np.clip(sleep_score, 0, 100)

    # Steps: ~6000/day with noise
    steps = np.repeat(np.random.normal(6000, 2500, DAYS), SAMPLES_PER_DAY)
    steps = np.clip(steps, 0, 25000)

    # Adherence: 0-1
    adherence = np.repeat(np.random.beta(7, 2, DAYS), SAMPLES_PER_DAY)

    # --- Label: deterioration in next 24h ---
    # Deterioration = glucose goes above 250 OR below 70 in next 288 samples
    labels = []
    for i in range(n):
        window_end = min(i + 288, n)
        future_glucose = glucose[i:window_end]
        deterioration = int(
            np.any(future_glucose > 250) or np.any(future_glucose < 70)
        )
        labels.append(deterioration)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "glucose": glucose,
        "hrv": hrv,
        "sleep_score": sleep_score,
        "steps": steps,
        "adherence": adherence,
        "label_24h": labels,
    })

    # Rolling z-scores (14-day window = 4032 samples)
    window = 4032
    for col in ["glucose", "hrv", "sleep_score", "steps", "adherence"]:
        roll_mean = df[col].rolling(window, min_periods=50).mean()
        roll_std  = df[col].rolling(window, min_periods=50).std().replace(0, 1)
        df[f"z_{col}"] = (df[col] - roll_mean) / roll_std

    return df

print("Generating synthetic T1DM dataset...")
all_patients = pd.concat([simulate_patient(i) for i in range(N_PATIENTS)])
out_path = Path("ml/synthetic_t1dm.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
all_patients.to_parquet(out_path, index=False)
print(f"Saved {len(all_patients):,} rows to {out_path}")
print(f"Deterioration rate: {all_patients['label_24h'].mean():.1%}")
