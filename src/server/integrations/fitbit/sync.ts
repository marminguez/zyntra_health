import { prisma } from "../../db/prisma";
import { encryptValue, decryptValue } from "../../security/crypto";
import { ingestSignal } from "../../zyntra/ingest";
import { refreshAccessToken } from "./oauth";
import { fetchDailySummary } from "./client";

/**
 * Syncs yesterday's Fitbit data for a patient.
 * Handles token refresh automatically.
 */
export async function syncFitbitForPatient(patientId: string, userId: string): Promise<{
  synced: string[];
  errors: string[];
}> {
  const token = await prisma.integrationToken.findUnique({
    where: { patientId_provider: { patientId, provider: "fitbit" } },
  });

  if (!token) throw new Error(`No Fitbit token for patient ${patientId}`);

  // Decrypt tokens
  let accessToken = await decryptValue(token.accessToken);
  let refreshToken = token.refreshToken ? await decryptValue(token.refreshToken) : null;

  // Refresh if expired (or within 5 min of expiry)
  if (token.expiresAt && token.expiresAt.getTime() < Date.now() + 5 * 60_000) {
    if (!refreshToken) throw new Error("Fitbit token expired and no refresh token available");
    const refreshed = await refreshAccessToken(refreshToken);
    accessToken = refreshed.accessToken;
    refreshToken = refreshed.refreshToken;

    // Persist refreshed tokens (encrypted)
    await prisma.integrationToken.update({
      where: { patientId_provider: { patientId, provider: "fitbit" } },
      data: {
        accessToken: await encryptValue(refreshed.accessToken),
        refreshToken: await encryptValue(refreshed.refreshToken),
        expiresAt:    refreshed.expiresAt,
      },
    });
  }

  // Fetch yesterday's data
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const dateStr = yesterday.toISOString().split("T")[0];

  const summary = await fetchDailySummary(accessToken, dateStr);
  const ts = `${dateStr}T23:59:00.000Z`;

  const synced: string[] = [];
  const errors: string[] = [];

  const signalsToIngest = [
    { type: "wearable_steps",            value: summary.steps,      unit: "steps" },
    { type: "wearable_hrv_ms",           value: summary.hrvRmssd,   unit: "ms"    },
    { type: "wearable_sleep_score",      value: summary.sleepScore, unit: "score" },
  ] as const;

  for (const signal of signalsToIngest) {
    if (signal.value === null) continue;
    try {
      await ingestSignal(
        { patientId, source: "WEARABLE", ts, type: signal.type, value: signal.value, unit: signal.unit },
        userId
      );
      synced.push(signal.type);
    } catch (err: unknown) {
      errors.push(`${signal.type}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  return { synced, errors };
}
