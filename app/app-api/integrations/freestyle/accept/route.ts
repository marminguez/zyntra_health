import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import { prisma } from "@/server/db/prisma";
import { emailsMatch } from "@/server/integrations/freestyle/onboarding";

export async function POST(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  const { patientId: requestedPatientId, acceptedEmail } = await req.json();
  const patientId = await resolvePatientIdForUser(auth.role, auth.userId, requestedPatientId);

  const connection = await prisma.libreConnection.findUnique({ where: { patientId } });
  if (!connection) return NextResponse.json({ error: "Start onboarding first." }, { status: 400 });

  const normalizedAcceptedEmail = String(acceptedEmail ?? "").trim().toLowerCase();
  if (!normalizedAcceptedEmail) {
    return NextResponse.json({ error: "Accepted email is required." }, { status: 400 });
  }

  const now = new Date();
  const matched = emailsMatch(connection.invitedEmail, normalizedAcceptedEmail);

  const updated = await prisma.libreConnection.update({
    where: { patientId },
    data: {
      acceptedEmail: normalizedAcceptedEmail,
      acceptedAt: now,
      lastCheckAt: now,
      status: matched ? "SHARE_ACCEPTED_NO_DATA_YET" : "EMAIL_MISMATCH",
      errorCode: matched ? "NO_GLUCOSE_UPLOADED_YET" : "EMAIL_MISMATCH",
      errorMessage: matched
        ? "The connection exists, but no glucose data has reached the cloud yet. Keep the Libre phone online and try again."
        : "The invitation must be accepted in LibreLinkUp with the same email used in Connected Apps.",
    },
  });

  console.info("[libre-onboarding] transition", {
    patientId,
    invitedEmail: connection.invitedEmail,
    acceptedEmail: normalizedAcceptedEmail,
    toStatus: updated.status,
    at: now.toISOString(),
  });

  return NextResponse.json({ status: updated.status, acceptedAt: updated.acceptedAt?.toISOString() ?? null });
}
