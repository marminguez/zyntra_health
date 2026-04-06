import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";
import { prisma } from "@/server/db/prisma";
import { encryptValue } from "@/server/security/crypto";
import { fetchLatestReadings } from "@/server/integrations/freestyle/client";

export async function POST(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  const { patientId: requestedPatientId, invitedEmail, librePassword, region } = await req.json();
  const patientId = await resolvePatientIdForUser(auth.role, auth.userId, requestedPatientId);

  const normalizedEmail = String(invitedEmail ?? "").trim().toLowerCase();
  if (!normalizedEmail) {
    return NextResponse.json({ error: "LibreLinkUp email is required." }, { status: 400 });
  }

  const now = new Date();
  let hasCredentials = false;
  let syncedSamples = 0;

  if (typeof librePassword === "string" && librePassword.trim()) {
    try {
      const readings = await fetchLatestReadings(normalizedEmail, librePassword);
      syncedSamples = readings.length;
      hasCredentials = true;

      await prisma.integrationToken.upsert({
        where: { patientId_provider: { patientId, provider: "freestyle" } },
        create: {
          patientId,
          provider: "freestyle",
          accessToken: await encryptValue(normalizedEmail),
          refreshToken: await encryptValue(librePassword.trim()),
        },
        update: {
          accessToken: await encryptValue(normalizedEmail),
          refreshToken: await encryptValue(librePassword.trim()),
        },
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Could not validate LibreLinkUp credentials";
      return NextResponse.json({ error: `Libre credentials are invalid or unavailable: ${message}` }, { status: 400 });
    }
  }

  const connection = await prisma.libreConnection.upsert({
    where: { patientId },
    create: {
      patientId,
      userId: auth.userId,
      invitedEmail: normalizedEmail,
      acceptedEmail: hasCredentials ? normalizedEmail : null,
      acceptedAt: hasCredentials ? now : null,
      status: hasCredentials ? (syncedSamples > 0 ? "SYNC_ACTIVE" : "WAITING_FOR_DATA") : "INVITE_SENT",
      inviteSentAt: now,
      lastCheckAt: now,
      diagnostics: JSON.stringify({ region: region ?? null, authMode: hasCredentials ? "DIRECT_LOGIN" : "FOLLOWER_SHARE" }),
      errorCode: hasCredentials && syncedSamples === 0 ? "NO_GLUCOSE_UPLOADED_YET" : null,
      errorMessage:
        hasCredentials && syncedSamples === 0
          ? "Credentials are valid but no glucose data is available yet. Keep Libre app online and retry."
          : null,
    },
    update: {
      userId: auth.userId,
      invitedEmail: normalizedEmail,
      acceptedEmail: hasCredentials ? normalizedEmail : null,
      acceptedAt: hasCredentials ? now : null,
      firstDataAt: null,
      lastDataAt: null,
      status: hasCredentials ? (syncedSamples > 0 ? "SYNC_ACTIVE" : "WAITING_FOR_DATA") : "INVITE_SENT",
      inviteSentAt: now,
      lastCheckAt: now,
      errorCode: hasCredentials && syncedSamples === 0 ? "NO_GLUCOSE_UPLOADED_YET" : null,
      errorMessage:
        hasCredentials && syncedSamples === 0
          ? "Credentials are valid but no glucose data is available yet. Keep Libre app online and retry."
          : null,
      diagnostics: JSON.stringify({ region: region ?? null, authMode: hasCredentials ? "DIRECT_LOGIN" : "FOLLOWER_SHARE" }),
    },
  });

  console.info("[libre-onboarding] transition", {
    patientId,
    invitedEmail: normalizedEmail,
    toStatus: connection.status,
    at: now.toISOString(),
  });

  return NextResponse.json({
    ok: true,
    status: connection.status,
    inviteSentAt: connection.inviteSentAt?.toISOString() ?? null,
    acceptedAt: connection.acceptedAt?.toISOString() ?? null,
    syncedSamples,
    mode: hasCredentials ? "DIRECT_LOGIN" : "FOLLOWER_SHARE",
  });
}
