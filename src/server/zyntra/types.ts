export type ZyntraStatus = "stable" | "unstable" | "deteriorating";
export type ZyntraTrend = "improving" | "worsening" | "neutral";
export type ZyntraConfidence = "low" | "medium" | "high";

export interface ZyntraInputFeatures {
  glucoseVariability: number; // CV % (e.g. 25 for 25%)
  timeInRange: number; // % (e.g. 80 for 80%)
  timeInRangeTrend: number; // change from baseline (e.g. -5 for 5% drop)
  sleepScore: number; // 0-100
  hrv: number; // ms
  activityMinutes: number; // minutes of activity
}

export interface ZyntraBaseline {
  glucoseVariabilityMean: number;
  glucoseVariabilityStd: number;
  timeInRangeMean: number;
  timeInRangeStd: number;
}

export interface ZyntraOutput {
  status: ZyntraStatus;
  trend: ZyntraTrend;
  riskScore: number; // 0-100
  confidence: ZyntraConfidence;
  explanation: string;
}
