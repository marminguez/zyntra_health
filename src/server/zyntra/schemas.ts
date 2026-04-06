import { z } from "zod";

export const SIGNAL_TYPES = [
  "cgm_glucose_mgdl",
  "wearable_hrv_ms",
  "wearable_sleep_score",
  "wearable_steps",
  "wearable_activity_intensity",
  "manual_adherence_score",
] as const;

export type SignalType = typeof SIGNAL_TYPES[number];

export const SignalIngestSchema = z.object({
  patientId: z.string().cuid(),
  source: z.enum(["CGM", "WEARABLE", "MANUAL", "SYSTEM"]),
  ts: z.string().datetime(),
  type: z.enum(SIGNAL_TYPES),
  value: z.number().finite().optional(),
  unit: z.string().max(20).optional(),
  meta: z.record(z.unknown()).optional(),
});

export type SignalIngestInput = z.infer<typeof SignalIngestSchema>;

export const RiskRequestSchema = z.object({
  patientId: z.string().cuid(),
});

export type RiskRequestInput = z.infer<typeof RiskRequestSchema>;
