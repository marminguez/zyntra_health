import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import { prisma } from "@/server/db/prisma";
import { DIAGNOSTIC_MESSAGES, type LibreDiagnosticCode } from "@/server/integrations/freestyle/onboarding";

export async function GET(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  const requestedPatientId = req.nextUrl.searchParams.get("patientId") ?? undefined;
  const patientId = await resolvePatientIdForUser(auth.role, auth.userId, requestedPatientId);

  const email = process.env.LIBRE_EMAIL;

  if (email) {
    return NextResponse.json({
        status: "SYNC_ACTIVE",
        connection: {
        id: "mock-env-conn",
        userId: auth.userId,
        invitedEmail: email,
        acceptedEmail: email,
        status: "SYNC_ACTIVE",
        inviteSentAt: new Date().toISOString(),
        acceptedAt: new Date().toISOString(),
        firstDataAt: new Date().toISOString(),
        lastDataAt: new Date().toISOString(),
        lastCheckAt: new Date().toISOString(),
        errorCode: null,
        errorMessage: null,
        diagnostics: {},
        },
    });
  }

  const connection = await prisma.libreConnection.findUnique({ where: { patientId } });

  if (!connection) {
    return NextResponse.json({ status: "NOT_STARTED", connection: null });
  }

  return NextResponse.json({
    status: connection.status,
    connection: {
      id: connection.id,
      userId: connection.userId,
      invitedEmail: connection.invitedEmail,
      acceptedEmail: connection.acceptedEmail,
      status: connection.status,
      inviteSentAt: connection.inviteSentAt?.toISOString() ?? null,
      acceptedAt: connection.acceptedAt?.toISOString() ?? null,
      firstDataAt: connection.firstDataAt?.toISOString() ?? null,
      lastDataAt: connection.lastDataAt?.toISOString() ?? null,
      lastCheckAt: connection.lastCheckAt?.toISOString() ?? null,
      errorCode: connection.errorCode,
      errorMessage:
        connection.errorCode && connection.errorCode in DIAGNOSTIC_MESSAGES
          ? DIAGNOSTIC_MESSAGES[connection.errorCode as LibreDiagnosticCode]
          : connection.errorMessage,
      diagnostics: connection.diagnostics ? safeParseJson(connection.diagnostics) : {},
    },
  });
}

function safeParseJson(value: string): Record<string, unknown> {
  try {
    return JSON.parse(value) as Record<string, unknown>;
  } catch {
    return {};
  }
}
