import { LibreLinkUpClient } from "@diakem/libre-link-up-api-client";

const CLIENT_VERSIONS = ["4.12.0", "4.9.0", "4.8.0", "4.7.0"] as const;

export interface LibreReading {
  value: number;
  timestamp: Date;
  trend: string;
}

export class LibreSyncError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "LibreSyncError";
    this.status = status;
  }
}

function extractStatus(err: unknown): number | undefined {
  const maybeStatus = (err as any)?.response?.status;
  return typeof maybeStatus === "number" ? maybeStatus : undefined;
}

function normalizeLibreError(err: unknown): LibreSyncError {
  const status = extractStatus(err);

  if (status === 401 || status === 403) {
    return new LibreSyncError(
      "LibreLink rejected access (401/403). Verify your LibreLinkUp email/password and shared data settings.",
      status
    );
  }

  if (status === 429) {
    return new LibreSyncError("LibreLink rate-limited requests (429). Please retry in a few minutes.", status);
  }

  if (status === 430) {
    return new LibreSyncError(
      "LibreLink rejected the session request (430). Retry in a minute; if it persists, reconnect your LibreLinkUp account.",
      status
    );
  }

  const message = err instanceof Error ? err.message : "Could not fetch LibreLink data";

  if (message.toLowerCase().includes("reading 'token'") || message.toLowerCase().includes('reading "token"')) {
    return new LibreSyncError(
      "LibreLink did not return a valid session token. Reconnect and ensure shared data is enabled in LibreLinkUp.",
      status
    );
  }

  return new LibreSyncError(message, status);
}

function parseReadings(response: unknown): LibreReading[] {
  const payload = response as { data?: unknown };
  if (!payload?.data) return [];

  const connection = Array.isArray(payload.data) ? payload.data[0] : payload.data;
  const graphData: unknown[] = (connection as { graphData?: unknown[] })?.graphData ?? [];

  return graphData
    .map((r: unknown) => {
      const reading = r as { Value?: number; Timestamp?: string; TrendArrow?: number };
      return {
        value: reading.Value ?? 0,
        timestamp: new Date(reading.Timestamp ?? Date.now()),
        trend: trendLabel(reading.TrendArrow ?? 0),
      };
    })
    .filter((r) => r.value > 0)
    .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
}

export async function fetchLatestReadings(email: string, password: string): Promise<LibreReading[]> {
  let lastError: unknown;

  for (const clientVersion of CLIENT_VERSIONS) {
    try {
      const { read } = LibreLinkUpClient({ username: email, password, clientVersion });
      const response = await read();
      return parseReadings(response);
    } catch (err) {
      lastError = err;
      const status = extractStatus(err);

      if (status === 401 || status === 403 || status === 404 || status === 430 || status === 502 || status === 503) {
        continue;
      }

      throw normalizeLibreError(err);
    }
  }

  throw normalizeLibreError(lastError);
}

function trendLabel(arrow: number): string {
  const labels: Record<number, string> = {
    1: "RapidlyFalling",
    2: "Falling",
    3: "FortyFiveDown",
    4: "Flat",
    5: "FortyFiveUp",
    6: "Rising",
    7: "RapidlyRising",
  };
  return labels[arrow] ?? "Unknown";
}
