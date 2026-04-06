import { FITBIT_API } from "./config";

export interface FitbitDailySummary {
  date:       string;   // YYYY-MM-DD
  steps:      number | null;
  hrvRmssd:   number | null;  // nocturnal HRV in ms
  sleepScore: number | null;  // 0-100
  spO2:       number | null;  // %
}

async function fitbitGet(path: string, accessToken: string): Promise<unknown> {
  const res = await fetch(`${FITBIT_API}${path}`, {
    headers: { "Authorization": `Bearer ${accessToken}` },
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Fitbit API ${path} failed (${res.status}): ${err}`);
  }
  return res.json();
}

/** Fetches a daily summary of all signals Zyntra needs */
export async function fetchDailySummary(
  accessToken: string,
  date: string   // YYYY-MM-DD
): Promise<FitbitDailySummary> {
  if (accessToken === "mock_access_token") {
    return {
      date,
      steps: 8432,
      hrvRmssd: 45,
      sleepScore: 82,
      spO2: 98,
    };
  }

  const [activityData, hrvData, sleepData] = await Promise.allSettled([
    fitbitGet(`/activities/date/${date}.json`, accessToken),
    fitbitGet(`/hrv/date/${date}.json`, accessToken),
    fitbitGet(`/sleep/date/${date}.json`, accessToken),
  ]);

  // Steps
  let steps: number | null = null;
  if (activityData.status === "fulfilled") {
    const a = activityData.value as any;
    steps = a?.summary?.steps ?? null;
  }

  // HRV rmssd (nocturnal, measured during sleep)
  let hrvRmssd: number | null = null;
  if (hrvData.status === "fulfilled") {
    const h = hrvData.value as any;
    const daily = h?.hrv?.[0]?.value;
    hrvRmssd = daily?.dailyRmssd ?? daily?.rmssd ?? null;
  }

  // Sleep score
  let sleepScore: number | null = null;
  if (sleepData.status === "fulfilled") {
    const s = sleepData.value as any;
    sleepScore = s?.sleep?.[0]?.efficiency ?? null;
  }

  return { date, steps, hrvRmssd, sleepScore, spO2: null };
}
