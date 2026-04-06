import { computeBaseline } from "./baseline";
import { computeRiskScore } from "./riskModel";
import { buildExplanationBlock } from "./explanation";
import type { ZyntraInputFeatures, ZyntraOutput } from "./types";
import type { ZyntraScenario } from "./scenario";

export interface ZyntraEngineInput {
  features: ZyntraInputFeatures;
  historicalGlucoseVariability: number[];
  historicalTimeInRange: number[];
}

/**
 * The Zyntra Engine — orchestrates baseline, risk, and explanation.
 */
export function runZyntraEngine(input: ZyntraEngineInput): ZyntraOutput {
  const { features, historicalGlucoseVariability, historicalTimeInRange } = input;

  const baseline = computeBaseline(historicalGlucoseVariability, historicalTimeInRange);
  const breakdown = computeRiskScore(features, baseline);
  const explanationBlock = buildExplanationBlock(
    breakdown.riskScore,
    features.timeInRangeTrend,
    breakdown
  );

  return {
    riskScore: breakdown.riskScore,
    ...explanationBlock,
  };
}

/**
 * Mock input factory — used in dev and demo. Replace with real data fetch.
 */
export function getMockEngineInput(
  scenario: ZyntraScenario = "stable"
): ZyntraEngineInput {
  const baseHistory = Array.from({ length: 14 }, (_, i) => 22 + Math.sin(i) * 2);
  const tirHistory = Array.from({ length: 14 }, (_, i) => 75 + Math.cos(i) * 3);

  const scenarios: Record<ZyntraScenario, ZyntraInputFeatures> = {
    stable: {
      glucoseVariability: 22,
      timeInRange: 78,
      timeInRangeTrend: 2,
      sleepScore: 74,
      hrv: 58,
      activityMinutes: 45,
    },
    unstable: {
      glucoseVariability: 30,
      timeInRange: 62,
      timeInRangeTrend: -8,
      sleepScore: 48,
      hrv: 35,
      activityMinutes: 18,
    },
    deteriorating: {
      glucoseVariability: 38,
      timeInRange: 48,
      timeInRangeTrend: -15,
      sleepScore: 28,
      hrv: 22,
      activityMinutes: 5,
    },
  };

  return {
    features: scenarios[scenario],
    historicalGlucoseVariability: baseHistory,
    historicalTimeInRange: tirHistory,
  };
}
