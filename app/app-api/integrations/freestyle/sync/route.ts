import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import { verifyLibreConnection } from "@/server/integrations/freestyle/verifier";
import { prisma } from "@/server/db/prisma";
import { decryptValue } from "@/server/security/crypto";
import { syncFreestyleForPatient } from "@/server/integrations/freestyle/sync";
import { LibreSyncError } from "@/server/integrations/freestyle/client";

export async function POST(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "SERVICE", "PATIENT");
  if (!auth.authorized) return auth.response;

  try {
    const { patientId: requestedPatientId } = await req.json();
    const patientId = await resolvePatientIdForUser(auth.role, auth.userId, requestedPatientId);

    const email = process.env.LIBRE_EMAIL;
    const password = process.env.LIBRE_PASSWORD;

    if (!email || !password) {
        return NextResponse.json({ error: "Missing LIBRE_EMAIL or LIBRE_PASSWORD in .env" }, { status: 500 });
    }

    const syncResult = await syncFreestyleForPatient(patientId, auth.userId, email, password);

    return NextResponse.json({
      status: "SYNC_ACTIVE",
      synced: syncResult.synced,
      errors: syncResult.errors,
      acceptedAt: new Date().toISOString(),
      lastDataAt: new Date().toISOString(),
      lastCheckAt: new Date().toISOString(),
      errorCode: null,
      errorMessage: null,
    });
  } catch (err: unknown) {
    if (err instanceof LibreSyncError) {
      const status = err.status === 401 || err.status === 403 || err.status === 430 ? 400 : 502;
      return NextResponse.json({ error: err.message }, { status });
    }
    const message = err instanceof Error ? err.message : "Failed to verify FreeStyle connection";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
