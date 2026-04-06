// This script is for manual verification only.
// It bypasses the Next.js runtime to test core logic.

import { DataAgent } from './src/server/zyntra/hack/DataAgent';
import { RiskAgent } from './src/server/zyntra/hack/RiskAgent';
import { CoachAgent } from './src/server/zyntra/hack/CoachAgent';

async function testPipeline() {
  const participantId = '2302';
  console.log(`--- Testing ZyntraHack Pipeline for ${participantId} ---`);

  const dataAgent = DataAgent.getInstance();
  const data = await dataAgent.getParticipantData(participantId);

  if (!data) {
    console.error('Failed to load data.');
    return;
  }

  const riskAgent = new RiskAgent();
  const risk = riskAgent.calculateRisk(data);
  console.log('--- Risk Assessment ---');
  console.log(`Score: ${risk.riskScore}`);
  console.log(`Level: ${risk.riskLevel}`);
  console.log(`Factors: ${risk.contributingFactors.join(', ')}`);

  const coachAgent = new CoachAgent();
  const coach = coachAgent.generateCoachResponse(risk);
  
  // 4. Mimic API response structure
  const latestGlucose = data.glucose.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];
  const response = {
    participantId,
    latestSignals: {
      glucose: {
        value: Number(latestGlucose.value.toFixed(1)),
        unit: 'mg/dL',
        timestamp: latestGlucose.timestamp
      }
    },
    riskScore: risk.riskScore,
    riskLevel: risk.riskLevel,
    contributingFactors: risk.contributingFactors,
    explanation: risk.explanation,
    coachMessage: coach.coachMessage,
    voicePayload: coach.voicePayload,
    alertPayload: coach.alertPayload
  };

  console.log('--- Final JSON Sample ---');
  console.log(JSON.stringify(response, null, 2));
  console.log('--- Pipeline Test Complete ---');
}

testPipeline().catch(console.error);
