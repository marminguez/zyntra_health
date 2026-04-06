import { NextRequest, NextResponse } from 'next/server';
import { DataAgent } from '@/server/zyntra/hack/DataAgent';
import { RiskAgent } from '@/server/zyntra/hack/RiskAgent';
import { CoachAgent } from '@/server/zyntra/hack/CoachAgent';

// ── Real CSV timestamps for each demo scenario ───────────────────────────────
// These are verified data points from Participant 2302's CSV files.
// 'unstable' is a window where glucose moved significantly without hitting extremes.
const DEMO_SCENARIOS: Record<string, string> = {
  high:     '2023-09-04T04:46:00', // ~203 mg/dL  → HIGH scenario
  low:      '2023-09-09T14:18:00', // ~57  mg/dL  → LOW scenario
  unstable: '2023-09-07T10:00:00', // mid-range but rapid movement → UNSTABLE
  stable:   '2023-09-09T03:00:00', // ~113 mg/dL  → STABLE scenario
};

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const participantId = searchParams.get('participantId');

  if (!participantId) {
    return NextResponse.json({ error: 'Missing participantId' }, { status: 400 });
  }

  if (participantId !== '2302') {
    return NextResponse.json({ error: 'Demo restricted to participant 2302' }, { status: 403 });
  }

  try {
    const dataAgent = DataAgent.getInstance();
    const riskAgent = new RiskAgent();
    const coachAgent = new CoachAgent();

    // 1. Load data (lazy-loaded singleton — CSV parsed once, then cached)
    const data = await dataAgent.getParticipantData(participantId);
    if (!data) {
      return NextResponse.json({ error: 'Data not found for participant 2302' }, { status: 404 });
    }

    // 2. Scenario filtering — slice glucose to the chosen demo window
    const scenario = searchParams.get('scenario') ?? 'latest';
    let filteredData = data;

    if (DEMO_SCENARIOS[scenario]) {
      const targetTime = new Date(DEMO_SCENARIOS[scenario]).getTime();
      // Use glucose up to and including the target timestamp to show realistic trend
      const slicedGlucose = data.glucose.filter(g => g.timestamp.getTime() <= targetTime);
      filteredData = { ...data, glucose: slicedGlucose };
    }

    // 3. Calculate risk
    const riskResult = riskAgent.calculateRisk(filteredData);

    // 4. For named demo scenarios, override glucoseScenario so the UI/voice/alert
    //    always matches the intended demo — regardless of whether the data at that
    //    exact timestamp perfectly crosses a threshold (e.g. 160 mg/dL → HIGH predictive).
    const DEMO_SCENARIO_MAP: Record<string, 'low' | 'high' | 'unstable' | 'stable'> = {
      high:   'high',
      low:    'low',
      stable: 'stable',
    };
    if (DEMO_SCENARIO_MAP[scenario]) {
      riskResult.glucoseScenario = DEMO_SCENARIO_MAP[scenario];
    }

    // 5. Generate predictive coaching & alert payload (uses overridden glucoseScenario)
    const coachResponse = coachAgent.generateCoachResponse(riskResult);

    // 5. Latest signals summary
    const sortedGlucose = [...filteredData.glucose].sort(
      (a, b) => b.timestamp.getTime() - a.timestamp.getTime()
    );
    const latestGlucose = sortedGlucose[0];
    const latestActivity = [...data.activity].sort(
      (a, b) => b.timestamp.getTime() - a.timestamp.getTime()
    )[0];

    const latestSignals = {
      glucose: latestGlucose
        ? { value: Number(latestGlucose.value.toFixed(1)), unit: 'mg/dL', timestamp: latestGlucose.timestamp }
        : null,
      activity: latestActivity
        ? { steps: latestActivity.steps, timestamp: latestActivity.timestamp }
        : null,
      lastSync: new Date().toISOString(),
    };

    // 6. Return unified response
    return NextResponse.json({
      participantId,
      scenario,
      glucoseScenario: riskResult.glucoseScenario,
      latestSignals,
      riskScore: riskResult.riskScore,
      riskLevel: riskResult.riskLevel,
      contributingFactors: riskResult.contributingFactors,
      explanation: riskResult.explanation,
      ...coachResponse,
    });

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('[API ZyntraHack] Error:', message);
    return NextResponse.json({ error: 'Internal Server Error', message }, { status: 500 });
  }
}
