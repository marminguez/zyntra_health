import { ZyntraConfidence, ZyntraOutput, ZyntraStatus, ZyntraTrend } from "./types";
import { RiskBreakdown } from "./riskModel";

const RISK_HIGH_THRESHOLD = 70;
const RISK_MODERATE_THRESHOLD = 45;

/**
 * Determines the overall health status from the risk score.
 */
export function deriveStatus(riskScore: number): ZyntraStatus {
  if (riskScore >= RISK_HIGH_THRESHOLD) return "deteriorating";
  if (riskScore >= RISK_MODERATE_THRESHOLD) return "unstable";
  return "stable";
}

/**
 * Derives the metabolic trend from the time-in-range trend value.
 * A negative trend with high risk = worsening, positive = improving.
 */
export function deriveTrend(tirTrend: number, riskScore: number): ZyntraTrend {
  if (tirTrend < -5 || riskScore > RISK_HIGH_THRESHOLD) return "worsening";
  if (tirTrend > 3 && riskScore < RISK_MODERATE_THRESHOLD) return "improving";
  return "neutral";
}

/**
 * Determines confidence based on how many degenerate inputs there are.
 * If several signals converge on a finding, confidence is higher.
 */
export function deriveConfidence(breakdown: RiskBreakdown): ZyntraConfidence {
  const highSignals = [
    breakdown.glucoseVariabilityScore > 60,
    breakdown.timeInRangeTrendScore > 60,
    breakdown.sleepScore > 60,
    breakdown.hrvScore > 60,
    breakdown.activityScore > 60,
  ].filter(Boolean).length;

  if (highSignals >= 4) return "high";
  if (highSignals >= 2) return "medium";
  return "low";
}

/**
 * Returns the name of the top contributing risk factor.
 */
function topDriver(breakdown: RiskBreakdown): string {
  const scores: [string, number][] = [
    ["glucose variability", breakdown.glucoseVariabilityScore],
    ["declining time-in-range", breakdown.timeInRangeTrendScore],
    ["poor sleep quality", breakdown.sleepScore],
    ["low heart rate variability", breakdown.hrvScore],
    ["reduced physical activity", breakdown.activityScore],
  ];
  scores.sort((a, b) => b[1] - a[1]);
  return scores[0][0];
}

/**
 * Generates a human-readable explanation of the Zyntra output.
 */
export function generateExplanation(
  status: ZyntraStatus,
  trend: ZyntraTrend,
  breakdown: RiskBreakdown
): string {
  const driver = topDriver(breakdown);

  if (status === "deteriorating") {
    return `Your metabolic patterns are showing signs of instability. The main contributor is ${driver}. Zyntra has detected a trajectory that previously correlated with increased disruption. Monitoring closely is advised.`;
  }

  if (status === "unstable") {
    return `Some signals are outside your usual baseline, primarily driven by ${driver}. There's no immediate concern, but patterns are shifting. Maintaining routine sleep, activity, and nutrition can help stabilise.`;
  }

  if (trend === "improving") {
    return `Your metabolic indicators are trending positively. ${driver.charAt(0).toUpperCase() + driver.slice(1)} is within a healthy range. Keep up your current routine.`;
  }

  return `Your metabolic markers are within your personal baseline. Zyntra detected no significant deviation. Continue your current habits for continued stability.`;
}

/**
 * Orchestrates building the full ZyntraOutput explanation block.
 */
export function buildExplanationBlock(
  riskScore: number,
  tirTrend: number,
  breakdown: RiskBreakdown
): Pick<ZyntraOutput, "status" | "trend" | "confidence" | "explanation"> {
  const status = deriveStatus(riskScore);
  const trend = deriveTrend(tirTrend, riskScore);
  const confidence = deriveConfidence(breakdown);
  const explanation = generateExplanation(status, trend, breakdown);

  return { status, trend, confidence, explanation };
}
