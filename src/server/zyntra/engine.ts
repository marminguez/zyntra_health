/**
 * Zyntra Risk Engine
 *
 * Computes a composite metabolic risk score from z-scores of key signals,
 * weighted and adjusted by prediction horizon.
 */

export interface ZScores {
    glucose?: number;
    sleep?: number;
    hrv?: number;
    activity?: number;
    adherence?: number;
}

export interface RiskResult {
    score: number;    // 0..1
    level: "low" | "medium" | "high";
    explanation: string;
    confidence: number; // 0..1 — fracción del peso total disponible
}

/** Weights per domain (must sum to 1). */
const WEIGHTS: Record<keyof ZScores, number> = {
    glucose: 0.35,
    sleep: 0.2,
    hrv: 0.2,
    activity: 0.15,
    adherence: 0.10,
};

/** Horizon decay factors. */
const HORIZON_FACTOR: Record<number, number> = {
    24: 1.0,
    48: 0.93,
    72: 0.88,
};

/**
 * Convert a z-score into a risk contribution [0..1].
 *
 * - For glucose: a positive z (above mean) increases risk.
 * - For sleep, hrv, activity: a negative z (below mean) increases risk.
 * - Adherence: negative z increases risk (lower adherence → higher risk).
 */
function zToContribution(domain: keyof ZScores, z: number): number {
  const raw = domain === "glucose" ? z : -z;
  return 1 / (1 + Math.exp(-raw * 0.8));
}

/**
 * Computes composite metabolic risk for a given horizon.
 *
 * @param zScores  z-scores per domain (optional keys if data unavailable)
 * @param horizonHrs  24, 48, or 72
 */
export const VALID_HORIZONS = [24, 48, 72] as const;
export type HorizonHrs = typeof VALID_HORIZONS[number];

export function computeRisk(zScores: ZScores, horizonHrs: HorizonHrs): RiskResult {
    const hFactor = HORIZON_FACTOR[horizonHrs] ?? 1.0;

    let totalWeight = 0;
    let weightedSum = 0;

    for (const [domain, weight] of Object.entries(WEIGHTS)) {
        const z = zScores[domain as keyof ZScores];
        if (z === undefined) continue;

        const contribution = zToContribution(domain as keyof ZScores, z);
        weightedSum += contribution * weight;
        totalWeight += weight;
    }

    const allContributions = (Object.entries(WEIGHTS) as [keyof ZScores, number][])
      .filter(([d]) => zScores[d] !== undefined)
      .map(([d, w]) => ({
        domain: d,
        contribution: zToContribution(d, zScores[d]!),
        weight: w,
      }))
      .sort((a, b) => b.contribution * b.weight - a.contribution * a.weight);

    const drivers = allContributions
      .filter(c => c.contribution > 0.55)
      .map(c => `${c.domain}(z=${zScores[c.domain]!.toFixed(2)})`);

    const topSignal = allContributions[0];
    const explanation = allContributions.length === 0
      ? `No signal data available (horizon ${horizonHrs}h)`
      : drivers.length > 0
        ? `Risk driven by: ${drivers.join(", ")} (horizon ${horizonHrs}h)`
        : `Elevated risk from ${topSignal.domain} and ${allContributions.length - 1} other signal(s) (horizon ${horizonHrs}h)`;

    // Normalise by available weight
    const rawScore = totalWeight > 0 ? weightedSum / totalWeight : 0;
    const score = Math.max(0, Math.min(1, rawScore * hFactor));

    let level: "low" | "medium" | "high";
    if (score < 0.33) level = "low";
    else if (score < 0.66) level = "medium";
    else level = "high";

    return { score, level, explanation, confidence: totalWeight };
}

import { execSync } from "child_process";
import path from "path";

export interface MLRiskResult {
  score:        number;
  level:        "low" | "medium" | "high";
  confidence:   number;
  modelVersion: string;
  featuresUsed: string[];
  fallback:     boolean;
  fallbackResult?: RiskResult;
}

/**
 * Calls the Python ML model via subprocess.
 * Falls back to computeRisk() if the model is not available.
 */
export function computeRiskML(zScores: ZScores, horizonHrs: HorizonHrs): MLRiskResult {
  const input = JSON.stringify({
    z_glucose:     zScores.glucose     ?? null,
    z_hrv:         zScores.hrv         ?? null,
    z_sleep_score: zScores.sleep       ?? null,
    z_steps:       zScores.activity    ?? null,
    z_adherence:   zScores.adherence   ?? null,
  });

  const modelPath = process.env.ML_MODEL_PATH ?? "ml/model.pkl";

  try {
    const scriptPath = path.resolve("ml/predict.py");
    // Handle Windows Python executable
    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    const output = execSync(`${pythonCmd} "${scriptPath}"`, {
      input,
      encoding: "utf-8",
      timeout:  5000,
      env:      { ...process.env, MODEL_PATH: modelPath },
    });

    const result = JSON.parse(output.trim()) as {
      score:         number;
      confidence:    number;
      model_version: string;
      features_used: string[];
      fallback:      boolean;
      error?:        string;
    };

    if (result.fallback || result.error) {
      const fallback = computeRisk(zScores, horizonHrs);
      return { ...fallback, modelVersion: "rule-based-fallback", featuresUsed: [], fallback: true, fallbackResult: fallback };
    }

    // Apply horizon decay to ML score
    const HORIZON_FACTOR: Record<number, number> = { 24: 1.0, 48: 0.93, 72: 0.88 };
    const decayedScore = Math.max(0, Math.min(1, result.score * (HORIZON_FACTOR[horizonHrs] ?? 1.0)));

    let level: "low" | "medium" | "high";
    if (decayedScore < 0.33) level = "low";
    else if (decayedScore < 0.66) level = "medium";
    else level = "high";

    return {
      score:        decayedScore,
      level,
      confidence:   result.confidence,
      modelVersion: result.model_version,
      featuresUsed: result.features_used,
      fallback:     false,
    };
  } catch {
    // Python not available or model error — fall back to rule engine
    const fallback = computeRisk(zScores, horizonHrs);
    return { ...fallback, modelVersion: "rule-based-fallback", featuresUsed: [], fallback: true, fallbackResult: fallback };
  }
}
