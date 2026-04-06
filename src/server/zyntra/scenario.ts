export const ZYNTRA_SCENARIOS = ["stable", "unstable", "deteriorating"] as const;

export type ZyntraScenario = (typeof ZYNTRA_SCENARIOS)[number];

export function isZyntraScenario(value: unknown): value is ZyntraScenario {
  return typeof value === "string" && (ZYNTRA_SCENARIOS as readonly string[]).includes(value);
}

export function parseZyntraScenario(value: unknown, fallback: ZyntraScenario = "stable"): ZyntraScenario {
  return isZyntraScenario(value) ? value : fallback;
}
