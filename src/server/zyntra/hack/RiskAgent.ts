import { ParticipantData } from './DataAgent';

export interface RiskResult {
  riskScore: number;
  riskLevel: 'low' | 'medium' | 'high';
  /** Detected metabolic scenario — drives messaging in CoachAgent */
  glucoseScenario: 'low' | 'high' | 'unstable' | 'stable';
  contributingFactors: string[];
  explanation: string;
}

export class RiskAgent {
  public calculateRisk(data: ParticipantData): RiskResult {
    let score = 20; // baseline

    const glucose = data.glucose.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    const latestGlucose = glucose[0]?.value ?? 100;
    const previousGlucose = glucose[1]?.value ?? 100;
    const trend = latestGlucose - previousGlucose; // mg/dL change between last two readings

    // ── 1. Hypo / Hyper (most important signal) ──────────────────────────
    let glucoseScenario: 'low' | 'high' | 'unstable' | 'stable' = 'stable';

    if (latestGlucose < 70) {
      score += 35;
      glucoseScenario = 'low';
    } else if (latestGlucose > 180) {
      score += 30;
      glucoseScenario = 'high';
    }

    // ── 2. Rapid trend ──────────────────────────────────────────────────
    const isRapidChange = Math.abs(trend) > 15;
    if (isRapidChange) {
      score += 15;
      if (glucoseScenario === 'stable') {
        // No extreme value but rapid movement → unstable
        glucoseScenario = 'unstable';
      }
    }

    // ── 3. 24-hour variability ──────────────────────────────────────────
    const last24h = glucose.filter(
      g => (new Date().getTime() - g.timestamp.getTime()) < 24 * 60 * 60 * 1000
    );
    if (last24h.length > 5) {
      const values = last24h.map(g => g.value);
      const range = Math.max(...values) - Math.min(...values);
      if (range > 80) {
        score += 20;
        if (glucoseScenario === 'stable') glucoseScenario = 'unstable';
      }
    }

    // ── 4. Activity ─────────────────────────────────────────────────────
    const recentSteps = data.activity
      .filter(a => (new Date().getTime() - a.timestamp.getTime()) < 24 * 60 * 60 * 1000)
      .reduce((sum, a) => sum + a.steps, 0);
    if (recentSteps < 3000) score += 10;

    // ── 5. Sleep ────────────────────────────────────────────────────────
    const latestSleep = data.sleep.sort((a, b) => b.endTime.getTime() - a.endTime.getTime())[0];
    if (latestSleep && latestSleep.durationMs < 6 * 60 * 60 * 1000) score += 10;

    // ── 6. Nutrition/Insulin ─────────────────────────────────────────────
    const recentCarbs = data.nutrition
      .filter(n => (new Date().getTime() - n.timestamp.getTime()) < 4 * 60 * 60 * 1000)
      .reduce((sum, n) => sum + n.carbs, 0);
    const recentBolus = data.insulin
      .filter(i => i.type === 'bolus' && (new Date().getTime() - i.timestamp.getTime()) < 4 * 60 * 60 * 1000)
      .reduce((sum, i) => sum + i.units, 0);
    if (recentCarbs > 60 && recentBolus < 2) score += 10;

    // ── Caps ─────────────────────────────────────────────────────────────
    const finalScore = Math.max(0, Math.min(100, score));
    let riskLevel: 'low' | 'medium' | 'high' = 'low';
    if (finalScore > 70) riskLevel = 'high';
    else if (finalScore > 30) riskLevel = 'medium';

    // ── Human-readable factors (max 3, no jargon) ─────────────────────
    const factors: string[] = [];
    if (latestGlucose < 70) factors.push('Low glucose');
    else if (latestGlucose > 180) factors.push('High glucose');
    if (isRapidChange) factors.push(trend > 0 ? 'Rising quickly' : 'Dropping quickly');
    if (last24h.length > 5) {
      const values = last24h.map(g => g.value);
      if (Math.max(...values) - Math.min(...values) > 80) factors.push('Swings in the last 24h');
    }
    if (recentSteps < 3000) factors.push('Low activity');
    if (recentCarbs > 60 && recentBolus < 2) factors.push('Recent high-carb meal');
    if (latestSleep && latestSleep.durationMs < 6 * 60 * 60 * 1000) factors.push('Short sleep');

    // ── Human-first explanation ─────────────────────────────────────────
    let explanation = 'Your glucose is steady and within your usual range.';
    if (glucoseScenario === 'low') {
      explanation = `Your glucose has dropped to ${Math.round(latestGlucose)} mg/dL — below the safe range.`;
    } else if (glucoseScenario === 'high') {
      explanation = `Your glucose is at ${Math.round(latestGlucose)} mg/dL — above your usual range.`;
    } else if (glucoseScenario === 'unstable') {
      explanation = `Your glucose is ${trend > 0 ? 'rising' : 'moving'} faster than usual, which may lead to instability.`;
    }

    return {
      riskScore: finalScore,
      riskLevel,
      glucoseScenario,
      contributingFactors: factors.length > 0 ? factors.slice(0, 3) : ['Steady rhythm'],
      explanation,
    };
  }
}
