export type RiskLevel = "low" | "medium" | "high";
export type EscalationFlag = "none" | "monitor" | "human_review";

export interface Driver {
    key: string;
    label: string;
    impact: "high" | "medium" | "low";
    direction: "up" | "down";
    explanation: string;
}

export interface Recommendation {
    id: string;
    title: string;
    description: string;
    type: "activity" | "meal_timing" | "sleep" | "monitoring" | "clinical_follow_up";
}

export interface PredictionResult {
    score48h: number;
    score72h: number;
    riskLevel48h: RiskLevel;
    riskLevel72h: RiskLevel;
    summary: string;
    predictedEvent: string;
    drivers: Driver[];
    recommendations: Recommendation[];
    escalationFlag: EscalationFlag;
}

// Realistic mock data for the 3 visual states

export const mockRiskLow: PredictionResult = {
    score48h: 12,
    score72h: 18,
    riskLevel48h: "low",
    riskLevel72h: "low",
    summary: "Your current pattern is stable.",
    predictedEvent: "Low risk of short-term deterioration",
    drivers: [
        { key: "sleep", label: "Restorative Sleep", impact: "high", direction: "up", explanation: "Consistent sleep architecture aligns with stable metabolic baselines." },
        { key: "activity", label: "Daily Steps", impact: "medium", direction: "up", explanation: "Adequate post-meal movement observed." }
    ],
    recommendations: [
        { id: "rec1", title: "Maintain current routine", description: "Your current lifestyle rhythms are supporting your metabolic stability perfectly.", type: "activity" }
    ],
    escalationFlag: "none"
};

export const mockRiskMedium: PredictionResult = {
    score48h: 45,
    score72h: 58,
    riskLevel48h: "medium",
    riskLevel72h: "medium",
    summary: "Your recent pattern is moving away from baseline.",
    predictedEvent: "Your pattern shows early signs of instability",
    drivers: [
        { key: "variability", label: "Higher recent glucose variability", impact: "high", direction: "up", explanation: "Fluctuations have increased by 15% vs your 14-day baseline." },
        { key: "sleep", label: "Sleep below usual pattern", impact: "medium", direction: "down", explanation: "2 hours less deep sleep detected last night." },
        { key: "activity", label: "Lower activity than baseline", impact: "low", direction: "down", explanation: "Daily step count is 30% below your usual average." }
    ],
    recommendations: [
        { id: "rec1", title: "Take a 15–20 minute walk after dinner", description: "Light movement helps clear post-meal glucose and can gently restore baseline stability.", type: "activity" },
        { id: "rec2", title: "Avoid a late high-carb dinner tonight", description: "Reducing late glycemic load can alleviate overnight variability.", type: "meal_timing" },
        { id: "rec3", title: "Prioritize sleep routine tonight", description: "Getting an extra hour of restorative sleep may improve morning stability.", type: "sleep" }
    ],
    escalationFlag: "monitor"
};

export const mockRiskHigh: PredictionResult = {
    score48h: 82,
    score72h: 88,
    riskLevel48h: "high",
    riskLevel72h: "high",
    summary: "Your pattern indicates significant deviation from baseline.",
    predictedEvent: "High likelihood of worsening metabolic stability",
    drivers: [
        { key: "recovery", label: "Worsening overnight recovery", impact: "high", direction: "down", explanation: "Elevated overnight baseline glucose combined with lower HRV." },
        { key: "postprandial", label: "Upward post-meal trend", impact: "high", direction: "up", explanation: "Prolonged clearance times detected over the last 3 meals." },
        { key: "stress", label: "Higher stress load", impact: "medium", direction: "up", explanation: "Continuous physiological stress signals from wearable." }
    ],
    recommendations: [
        { id: "rec1", title: "Monitor post-meal trends more closely for 24h", description: "Keep an eye on how different meals affect your levels today.", type: "monitoring" },
        { id: "rec2", title: "Reduce simple carbohydrates", description: "Favoring complex carbs and proteins naturally cushions glucose variations.", type: "meal_timing" }
    ],
    escalationFlag: "human_review"
};

// Default export uses medium to show off the UI well
export const currentRiskProfile: PredictionResult = mockRiskMedium;
