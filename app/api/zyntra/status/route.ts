import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/server/auth/auth";
import { runZyntraEngine, getMockEngineInput } from "@/server/zyntra/zyntraEngine";
import type { ZyntraEngineInput } from "@/server/zyntra/zyntraEngine";
import { parseZyntraScenario } from "@/server/zyntra/scenario";
import { buildLiveEngineInput, RealDataUnavailableError } from "@/server/zyntra/liveInput";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";

/**
 * GET /api/zyntra/status
 *
 * Returns the patient's current Zyntra risk profile.
 * Uses real patient signals (Libre + wearable).
 * Falls back to mock data only when real signals are unavailable.
 */
export async function GET(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const role = ((session.user as any)?.role ?? "PATIENT") as "ADMIN" | "CLINICIAN" | "PATIENT" | "SERVICE";
    const userId = (session.user as any)?.id as string;
    const requestedPatientId = (session.user as any)?.patientId as string | undefined;
    const patientId = await resolvePatientIdForUser(role, userId, requestedPatientId);

    const { searchParams } = new URL(request.url);
    const scenario = parseZyntraScenario(searchParams.get("scenario"));

    let input: ZyntraEngineInput;
    let mode: "real" | "mock" = "real";

    try {
      input = await buildLiveEngineInput(patientId);
    } catch (err) {
      if (!(err instanceof RealDataUnavailableError)) throw err;
      input = getMockEngineInput(scenario);
      mode = "mock";
    }

    const result = runZyntraEngine(input);

    return NextResponse.json({
      ...result,
      generatedAt: new Date().toISOString(),
      dataMode: mode,
    });
  } catch (err) {
    console.error("[zyntra/status] Error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
