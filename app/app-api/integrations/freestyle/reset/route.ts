import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import { prisma } from "@/server/db/prisma";

export async function POST(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  const { patientId: requestedPatientId } = await req.json();
  const patientId = await resolvePatientIdForUser(auth.role, auth.userId, requestedPatientId);

  await prisma.libreConnection.deleteMany({ where: { patientId } });
  await prisma.integrationToken.deleteMany({ where: { patientId, provider: "freestyle" } });

  console.info("[libre-onboarding] reset", {
    patientId,
    at: new Date().toISOString(),
  });

  return NextResponse.json({ ok: true, status: "NOT_STARTED" });
}
