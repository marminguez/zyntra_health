import { RiskResult } from './RiskAgent';

// ─────────────────────────────────────────────────────────────────────────────
// PREDICTIVE SCENARIO TEMPLATES
// Each scenario has a strictly defined headline, action, voice, and alert.
// CRITICAL RULES:
//   - LOW  → NEVER suggest walking / movement
//   - HIGH → NEVER suggest eating
//   - UNSTABLE → no directional advice, focus on balance
//   - STABLE → positive reinforcement, no alarm
// ─────────────────────────────────────────────────────────────────────────────

const SCENARIO_TEMPLATES = {
  low: {
    primaryMessage: "You may go low in the next hour",
    secondaryMessage: "It's lower than your usual range",
    coachMessage: "You should eat something soon.",
    voiceText: "You may go low soon. You should eat something now.",
    alertTitle: "Glucose is Low",
    alertMessage: "Your glucose is low. You should eat something now.",
    lockscreenPreview: "Your glucose is low. Eat something soon.",
  },
  high: {
    primaryMessage: "Your glucose may rise further",
    secondaryMessage: "It's higher than your usual range",
    coachMessage: "A short walk could help bring it down.",
    voiceText: "Your glucose may keep rising. A short walk could help.",
    alertTitle: "Glucose is High",
    alertMessage: "Your glucose is high. A short walk could help bring it down.",
    lockscreenPreview: "Your glucose is high. A short walk could help.",
  },
  unstable: {
    primaryMessage: "Your glucose may become unstable",
    secondaryMessage: "It's moving outside your usual steady state",
    coachMessage: "Try to keep your next meal balanced.",
    voiceText: "Your glucose may fluctuate. Try to keep your next meal balanced.",
    alertTitle: "Glucose is Shifting",
    alertMessage: "Your glucose is shifting. Keep your next meal balanced.",
    lockscreenPreview: "Your glucose is shifting. Keep meals balanced.",
  },
  stable: {
    primaryMessage: "You're likely to stay stable",
    secondaryMessage: "Everything looks within your usual range",
    coachMessage: "Keep your current routine.",
    voiceText: "Everything looks stable right now.",
    alertTitle: "",
    alertMessage: "",
    lockscreenPreview: "",
  },
} as const;

export interface CoachResponse {
  primaryMessage: string;
  secondaryMessage: string;
  coachMessage: string;
  voicePayload: {
    voiceText: string;
    shouldSpeak: boolean;
  };
  alertPayload: {
    shouldAlert: boolean;
    alertTitle: string;
    alertMessage: string;
    alertSeverity: 'low' | 'medium' | 'high';
    lockscreenPreview: string;
    voiceText: string;
  };
}

export class CoachAgent {
  public generateCoachResponse(risk: RiskResult): CoachResponse {
    const scenario = risk.glucoseScenario;
    const tmpl = SCENARIO_TEMPLATES[scenario];

    // ── Alert logic ──────────────────────────────────────────────────────
    // Low and High always alert. Unstable only alerts on high risk score.
    // Stable never alerts.
    const shouldAlert =
      scenario === 'low' ||
      scenario === 'high' ||
      (scenario === 'unstable' && risk.riskLevel === 'high');

    return {
      primaryMessage: tmpl.primaryMessage,
      secondaryMessage: tmpl.secondaryMessage,
      coachMessage: tmpl.coachMessage,
      voicePayload: {
        voiceText: tmpl.voiceText,
        shouldSpeak: scenario !== 'stable',
      },
      alertPayload: {
        shouldAlert,
        alertTitle: tmpl.alertTitle,
        alertMessage: tmpl.alertMessage,
        alertSeverity: risk.riskLevel,
        lockscreenPreview: tmpl.lockscreenPreview,
        voiceText: tmpl.voiceText,
      },
    };
  }
}
