import { ZyntraBaseline } from "./types";

/**
 * Computes mean of an array of numbers.
 */
function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Computes population standard deviation of an array of numbers.
 */
function std(values: number[], mu?: number): number {
  if (values.length < 2) return 0;
  const m = mu ?? mean(values);
  const variance = values.reduce((sum, v) => sum + Math.pow(v - m, 2), 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Computes a rolling (sliding window) mean and std for the last `window` entries.
 */
export function rollingStats(
  values: number[],
  window: number
): { mean: number; std: number } {
  const slice = values.slice(-window);
  const m = mean(slice);
  return { mean: m, std: std(slice, m) };
}

/**
 * Computes the Zyntra baseline from historical glucose variability and time-in-range arrays.
 * Uses a 14-day (last 14 entries) rolling window.
 */
export function computeBaseline(
  historicalGlucoseVariability: number[],
  historicalTimeInRange: number[],
  windowDays = 14
): ZyntraBaseline {
  const gv = rollingStats(historicalGlucoseVariability, windowDays);
  const tir = rollingStats(historicalTimeInRange, windowDays);

  return {
    glucoseVariabilityMean: gv.mean,
    glucoseVariabilityStd: gv.std,
    timeInRangeMean: tir.mean,
    timeInRangeStd: tir.std,
  };
}

/**
 * Computes a z-score indicating how many standard deviations a current value
 * is from a given mean, clamped to [-3, 3].
 */
export function zScore(current: number, baseline: number, stdDev: number): number {
  if (stdDev === 0) return 0;
  return Math.max(-3, Math.min(3, (current - baseline) / stdDev));
}
