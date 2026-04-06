import { prisma } from "@/server/db/prisma";
import {
  DIAGNOSTIC_MESSAGES,
  evaluateConnectionState,
  type LibreConnectionStatus,
} from "./onboarding";

type VerifyParams = {
  patientId: string;
  userId: string;
  isOnline?: boolean;
  region?: string | null;
};

export async function verifyLibreConnection(params: VerifyParams) {
  const connection = await prisma.libreConnection.findUnique({ where: { patientId: params.patientId } });
  const now = new Date();

  if (!connection) {
    throw new Error("Libre connection not initialized");
  }

  const latestSignal = await prisma.signal.findFirst({
    where: {
      patientId: params.patientId,
      type: "cgm_glucose_mgdl",
      source: "CGM",
    },
    orderBy: { ts: "desc" },
    select: { ts: true },
  });

  const result = evaluateConnectionState({
    invitedEmail: connection.invitedEmail,
    acceptedEmail: connection.acceptedEmail,
    acceptedAt: connection.acceptedAt,
    hasAnyData: Boolean(latestSignal?.ts),
    lastDataAt: latestSignal?.ts ?? null,
    isOnline: params.isOnline,
    regionMatches: connection.diagnostics ? parseRegionMatch(connection.diagnostics, params.region) : true,
  });

  const nextStatus: LibreConnectionStatus =
    result.status === "WAITING_FOR_LIBRELINKUP_ACCEPTANCE" && connection.status === "INVITE_SENT"
      ? "WAITING_FOR_LIBRELINKUP_ACCEPTANCE"
      : result.status;

  const updated = await prisma.libreConnection.update({
    where: { patientId: params.patientId },
    data: {
      status: nextStatus,
      lastCheckAt: now,
      lastDataAt: latestSignal?.ts ?? null,
      firstDataAt: connection.firstDataAt ?? latestSignal?.ts ?? null,
      errorCode: result.errorCode,
      errorMessage: result.errorMessage,
      diagnostics: JSON.stringify({
        ...safeParseJson(connection.diagnostics),
        verifierResult: result,
        userVisibleMessage: result.errorCode ? DIAGNOSTIC_MESSAGES[result.errorCode] : null,
        region: params.region ?? null,
      }),
    },
  });

  console.info("[libre-onboarding] verifier", {
    patientId: params.patientId,
    invitedEmail: connection.invitedEmail,
    acceptedEmail: connection.acceptedEmail,
    fromStatus: connection.status,
    toStatus: updated.status,
    at: now.toISOString(),
    verifierResult: result.status,
    errorCode: result.errorCode,
  });

  return updated;
}

function parseRegionMatch(diagnosticsRaw: string, region: string | null | undefined): boolean {
  const diagnostics = safeParseJson(diagnosticsRaw);
  const invitedRegion = typeof diagnostics.region === "string" ? diagnostics.region : null;
  if (!invitedRegion || !region) return true;
  return invitedRegion.toLowerCase() === region.toLowerCase();
}

function safeParseJson(value: string | null): Record<string, unknown> {
  if (!value) return {};
  try {
    return JSON.parse(value) as Record<string, unknown>;
  } catch {
    return {};
  }
}
