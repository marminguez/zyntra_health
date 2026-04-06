import { prisma } from "@/server/db/prisma";
import { decryptNumber } from "@/server/security/crypto";
import type { ZyntraEngineInput } from "./zyntraEngine";

const TIR_MIN = 70;
const TIR_MAX = 180;
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const LOOKBACK_DAYS = 14;

type SignalRow = {
  type: string;
  ts: Date;
  value: number | null;
  encryptedValue: string | null;
};

export class RealDataUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RealDataUnavailableError";
  }
}

function dayKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const mu = mean(values);
  const variance = values.reduce((acc, value) => acc + (value - mu) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

async function resolveNumericValue(row: SignalRow): Promise<number | null> {
  if (row.value !== null) return row.value;
  if (!row.encryptedValue) return null;

  try {
    return await decryptNumber(row.encryptedValue);
  } catch {
    return null;
  }
}

async function getCgmValuesByDay(patientId: string): Promise<Map<string, number[]>> {
  const since = new Date(Date.now() - LOOKBACK_DAYS * MS_PER_DAY);
  const cgmRows = await prisma.signal.findMany({
    where: {
      patientId,
      type: "cgm_glucose_mgdl",
      ts: { gte: since },
    },
    orderBy: { ts: "asc" },
    select: { type: true, ts: true, value: true, encryptedValue: true },
  });

  const byDay = new Map<string, number[]>();

  for (const row of cgmRows) {
    const value = await resolveNumericValue(row);
    if (value === null || !Number.isFinite(value) || value <= 20 || value > 500) continue;

    const key = dayKey(row.ts);
    const existing = byDay.get(key) ?? [];
    existing.push(value);
    byDay.set(key, existing);
  }

  return byDay;
}

function computeDailyMetrics(values: number[]): { cv: number; tir: number } {
  const mu = mean(values);
  const sigma = std(values);
  const cv = mu > 0 ? (sigma / mu) * 100 : 0;
  const inRange = values.filter((value) => value >= TIR_MIN && value <= TIR_MAX).length;
  const tir = (inRange / values.length) * 100;

  return {
    cv: clamp(cv, 0, 80),
    tir: clamp(tir, 0, 100),
  };
}

async function latestWearableMetric(patientId: string, type: string, fallback: number): Promise<number> {
  const row = await prisma.signal.findFirst({
    where: { patientId, type },
    orderBy: { ts: "desc" },
    select: { type: true, ts: true, value: true, encryptedValue: true },
  });

  if (!row) return fallback;
  const value = await resolveNumericValue(row);
  return value === null || !Number.isFinite(value) ? fallback : value;
}

export async function buildLiveEngineInput(patientId: string): Promise<ZyntraEngineInput> {
  const cgmByDay = await getCgmValuesByDay(patientId);

  if (cgmByDay.size === 0) {
    throw new RealDataUnavailableError("No CGM data found. Connect LibreView/LibreLink and run sync.");
  }

  const orderedDays = [...cgmByDay.entries()]
    .filter(([, values]) => values.length >= 12)
    .sort(([a], [b]) => a.localeCompare(b));

  if (orderedDays.length < 2) {
    throw new RealDataUnavailableError("Not enough CGM data points to compute trend yet.");
  }

  const dayMetrics = orderedDays.map(([, values]) => computeDailyMetrics(values));
  const latestMetrics = dayMetrics[dayMetrics.length - 1];

  const historicalWindow = dayMetrics.slice(-14);
  const historicalGlucoseVariability = historicalWindow.map((metric) => metric.cv);
  const historicalTimeInRange = historicalWindow.map((metric) => metric.tir);

  const baselineTir = mean(historicalTimeInRange.slice(0, Math.max(1, historicalTimeInRange.length - 1)));
  const timeInRangeTrend = latestMetrics.tir - baselineTir;

  const [sleepScore, hrv, steps] = await Promise.all([
    latestWearableMetric(patientId, "wearable_sleep_score", 70),
    latestWearableMetric(patientId, "wearable_hrv_ms", 45),
    latestWearableMetric(patientId, "wearable_steps", 7000),
  ]);

  return {
    features: {
      glucoseVariability: latestMetrics.cv,
      timeInRange: latestMetrics.tir,
      timeInRangeTrend,
      sleepScore: clamp(sleepScore, 0, 100),
      hrv: clamp(hrv, 10, 150),
      activityMinutes: clamp(steps / 100, 0, 120),
    },
    historicalGlucoseVariability,
    historicalTimeInRange,
  };
}
