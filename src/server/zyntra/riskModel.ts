import { ZyntraBaseline, ZyntraInputFeatures } from "./types";
import { zScore } from "./baseline";

// Feature weights — must sum to 1.0
const WEIGHTS = {
  glucoseVariability: 0.30,
  timeInRangeTrend: 0.25,
  sleep: 0.20,
  hrv: 0.15,
  activity: 0.10,
};

/**
 * Normalises a value from a source range to [0, 100].
 * If invert is true, higher input = higher risk (e.g., glucose variability).
 */
function toRiskScore(
  value: number,
  min: number,
  max: number,
  higherIsRiskier = true
): number {
  const clamped = Math.max(min, Math.min(max, value));
  const raw = (clamped - min) / (max - min); // 0-1
  const normalised = higherIsRiskier ? raw : 1 - raw;
  return normalised * 100;
}

export interface RiskBreakdown {
  glucoseVariabilityScore: number;
  timeInRangeTrendScore: number;
  sleepScore: number;
  hrvScore: number;
  activityScore: number;
  riskScore: number; // 0–100, higher = more risk
}

/**
 * Computes the composite risk score (0–100) from all feature inputs.
 *
 * Each sub-score is computed on a 0–100 scale where 100 = highest risk.
 */
export function computeRiskScore(
  features: ZyntraInputFeatures,
  baseline: ZyntraBaseline
): RiskBreakdown {
  // Glucose Variability: CV > 36% is very high risk.
  const gvZ = zScore(
    features.glucoseVariability,
    baseline.glucoseVariabilityMean,
    baseline.glucoseVariabilityStd
  );
  // Map z-score from [-3,3] → deviation score [0,100] where +3 = 100 risk
  const glucoseVariabilityScore = toRiskScore(gvZ + 3, 0, 6, true);

  // Time in Range Trend: negative trend is bad.
  const tirTrendScore = toRiskScore(features.timeInRangeTrend + 20, 0, 40, false);

  // Sleep: 0 = no sleep (worst), 100 = perfect
  const sleepScore = toRiskScore(features.sleepScore, 0, 100, false);

  // HRV: low HRV is bad.
  const hrvScore = toRiskScore(features.hrv, 10, 100, false);

  // Activity: sedentary is bad.
  const activityScore = toRiskScore(features.activityMinutes, 0, 90, false);

  const riskScore =
    glucoseVariabilityScore * WEIGHTS.glucoseVariability +
    tirTrendScore * WEIGHTS.timeInRangeTrend +
    sleepScore * WEIGHTS.sleep +
    hrvScore * WEIGHTS.hrv +
    activityScore * WEIGHTS.activity;

  return {
    glucoseVariabilityScore: Math.round(glucoseVariabilityScore),
    timeInRangeTrendScore: Math.round(tirTrendScore),
    sleepScore: Math.round(sleepScore),
    hrvScore: Math.round(hrvScore),
    activityScore: Math.round(activityScore),
    riskScore: Math.round(Math.max(0, Math.min(100, riskScore))),
  };
}
