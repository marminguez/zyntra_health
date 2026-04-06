import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/server/auth/auth";
import { runZyntraEngine, getMockEngineInput } from "@/server/zyntra/zyntraEngine";
import type { ZyntraOutput, ZyntraStatus } from "@/server/zyntra/types";
import { parseZyntraScenario } from "@/server/zyntra/scenario";
import { buildLiveEngineInput, RealDataUnavailableError } from "@/server/zyntra/liveInput";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import type { ZyntraEngineInput } from "@/server/zyntra/zyntraEngine";

const ALERT_THRESHOLD = 70;
const STATUS_REPLIES: Record<ZyntraStatus, string> = {
  stable: "You're doing well. Your metabolic patterns are within your personal baseline.",
  unstable:
    "Some of your signals are outside your usual range right now. Nothing alarming, but worth being mindful of.",
  deteriorating:
    "Zyntra has detected a pattern that previously led to instability. It's a good time to pay closer attention to sleep, movement, and meals.",
};

// ── Scenario-aware replies for ZyntraHack demo ───────────────────────────────

interface ScenarioContext {
  glucoseScenario?: string | null;
  primaryMessage?: string | null;
  coachMessage?: string | null;
  explanation?: string | null;
  riskScore?: number | null;
}

function buildScenarioReply(query: string, ctx: ScenarioContext): string | null {
  const s = ctx.glucoseScenario;
  if (!s) return null; // No scenario context → fall through to legacy engine

  const q = query.toLowerCase().trim();

  const isStatus = q.includes("how am i") || q.includes("doing") || q.includes("status") || q.includes("what") && q.includes("now");
  const isWhy    = q.includes("why") || q.includes("reason") || q.includes("explain") || q.includes("because");
  const isAction = q.includes("what can i do") || q.includes("what should") || q.includes("action") || q.includes("help") || q.includes("do now") || q.includes("do?");

  if (s === "low") {
    if (isStatus)  return ctx.primaryMessage
      ? `${ctx.primaryMessage}. Your glucose is trending downward and may drop below the safe range within the next hour. You need to act now.`
      : "Your glucose may go low soon. You need to eat something now before it drops further.";
    if (isWhy)    return ctx.explanation
      ? `${ctx.explanation} When glucose drops this low this quickly, your body doesn't have enough energy to function normally — that's why it's important to act before you feel symptoms.`
      : "Your glucose is dropping quickly. If you don't eat something now, it could fall below the safe range and cause symptoms like shakiness or dizziness.";
    if (isAction) return "Eat something with fast-acting carbs right now — a small glass of juice, a few glucose tablets, or a piece of fruit. Avoid waiting. Check again in 15 minutes.";
    return `${ctx.primaryMessage ?? "Your glucose may go low soon"}. ${ctx.coachMessage ?? "You should eat something now."} Is there anything specific you'd like to know?`;
  }

  if (s === "high") {
    if (isStatus)  return ctx.primaryMessage
      ? `${ctx.primaryMessage}. Your glucose is at ${ctx.riskScore ? "an elevated level" : "a higher level"} and trending upward. Acting now can prevent it from going higher.`
      : "Your glucose may keep rising. A short walk could help bring it down before it goes higher.";
    if (isWhy)    return ctx.explanation
      ? `${ctx.explanation} When glucose rises without intervention, it can continue climbing — especially if you've had a recent meal or lower activity.`
      : "Your glucose is trending upward. Without action, it may continue rising beyond the normal range.";
    if (isAction) return "A short 10–15 minute walk is one of the most effective ways to bring glucose down naturally. Avoid eating high-carb foods right now, and stay hydrated.";
    return `${ctx.primaryMessage ?? "Your glucose may rise further"}. ${ctx.coachMessage ?? "A short walk could help."} Anything else you'd like to know?`;
  }

  if (s === "stable") {
    if (isStatus)  return "Your glucose looks stable right now — everything is within your usual range. Zyntra isn't detecting any concerning patterns at the moment.";
    if (isWhy)    return "Your recent glucose readings, activity, and sleep are all within normal patterns. No significant variability or trend detected.";
    if (isAction) return "Keep your current routine. Consistent sleep, regular light activity, and balanced meals are what's keeping you stable. No changes needed right now.";
    return "Everything looks stable. Your glucose is steady and within your usual range. Is there anything you'd like to check?";
  }

  return null;
}

function buildConversationReply(
  query: string,
  output: ZyntraOutput
): string {
  const q = query.toLowerCase().trim();

  // "How am I doing?"
  if (q.includes("how am i") || q.includes("doing") || q.includes("status")) {
    const trend =
      output.trend === "improving"
        ? " The good news: things appear to be trending in a positive direction."
        : output.trend === "worsening"
        ? " The trajectory is currently declining — small actions now may help."
        : "";
    return `${STATUS_REPLIES[output.status]}${trend} Your current risk score is ${output.riskScore}/100 (confidence: ${output.confidence}).`;
  }

  // "Why?"
  if (q.includes("why") || q.includes("reason") || q.includes("explain")) {
    return output.explanation;
  }

  // "What can I do?"
  if (
    q.includes("what can i do") ||
    q.includes("action") ||
    q.includes("help") ||
    q.includes("improve")
  ) {
    if (output.status === "stable") {
      return "Your patterns are stable — keep doing what you're doing. Consistent sleep, regular light activity, and balanced meals are your strongest tools.";
    }
    if (output.status === "unstable") {
      return "Focus on three things: aim for 7+ hours of sleep tonight, take a 20-minute walk, and avoid skipping meals. Small consistent actions compound quickly.";
    }
    return "Zyntra suggests prioritising sleep above all else right now — it's the single highest-impact signal. After that, a short walk and a regular meal schedule can help stabilise your patterns. If instability continues for more than 2 days, consider checking in with your care team.";
  }

  // Proactive alert message (triggered when riskScore is high)
  if (output.riskScore > ALERT_THRESHOLD) {
    return `You are following a pattern that previously led to instability. Your risk score is ${output.riskScore}/100. ${output.explanation}`;
  }

  // Fallback
  return `I'm here to help. You can ask me "How am I doing?", "Why?", or "What can I do?" to get a personalised insight based on your latest data.`;
}

/**
 * POST /api/zyntra/conversation
 * Body: { message: string; scenario?: "stable" | "unstable" | "deteriorating" }
 */
export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const role = ((session.user as any)?.role ?? "PATIENT") as "ADMIN" | "CLINICIAN" | "PATIENT" | "SERVICE";
    const userId = (session.user as any)?.id as string;
    const requestedPatientId = (session.user as any)?.patientId as string | undefined;
    const patientId = await resolvePatientIdForUser(role, userId, requestedPatientId);

    const body = await request.json();
    const message: string = body?.message ?? "";
    const scenario = parseZyntraScenario(body?.scenario);

    if (!message) {
      return NextResponse.json({ error: "message is required" }, { status: 400 });
    }

    // ── ZyntraHack: scenario-aware reply (takes priority) ────────────────────
    const scenarioCtx: ScenarioContext = {
      glucoseScenario: body?.glucoseScenario ?? null,
      primaryMessage:  body?.primaryMessage  ?? null,
      coachMessage:    body?.coachMessage    ?? null,
      explanation:     body?.explanation     ?? null,
      riskScore:       body?.riskScore       ?? null,
    };
    const scenarioReply = buildScenarioReply(message, scenarioCtx);
    if (scenarioReply) {
      return NextResponse.json({ reply: scenarioReply, dataMode: "scenario" });
    }

    // ── Legacy ZyntraEngine fallback ─────────────────────────────────────────
    let input: ZyntraEngineInput;
    let mode: "real" | "mock" = "real";

    try {
      input = await buildLiveEngineInput(patientId);
    } catch (err) {
      if (!(err instanceof RealDataUnavailableError)) throw err;
      input = getMockEngineInput(scenario);
      mode = "mock";
    }

    const output = runZyntraEngine(input);
    const reply = buildConversationReply(message, output);

    return NextResponse.json({
      reply,
      riskScore: output.riskScore,
      status: output.status,
      trend: output.trend,
      dataMode: mode,
    });
  } catch (err) {
    console.error("[zyntra/conversation] Error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
