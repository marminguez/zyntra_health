export const LIBRE_CONNECTION_STATUSES = [
  "NOT_STARTED",
  "INVITE_SENT",
  "WAITING_FOR_LIBRELINKUP_ACCEPTANCE",
  "SHARE_ACCEPTED_NO_DATA_YET",
  "WAITING_FOR_DATA",
  "SYNC_ACTIVE",
  "SYNC_ERROR",
  "EMAIL_MISMATCH",
  "NETWORK_OR_UPLOAD_DELAY",
] as const;

export type LibreConnectionStatus = (typeof LIBRE_CONNECTION_STATUSES)[number];

export const LIBRE_DIAGNOSTIC_CODES = [
  "EMAIL_MISMATCH",
  "INVITATION_NOT_ACCEPTED",
  "NO_GLUCOSE_UPLOADED_YET",
  "NETWORK_DELAY",
  "UNKNOWN",
] as const;

export type LibreDiagnosticCode = (typeof LIBRE_DIAGNOSTIC_CODES)[number];

export const DIAGNOSTIC_MESSAGES: Record<LibreDiagnosticCode, string> = {
  EMAIL_MISMATCH:
    "The invitation must be accepted in LibreLinkUp with the same email used in Connected Apps.",
  INVITATION_NOT_ACCEPTED:
    "The share invitation was sent, but it has not been accepted inside LibreLinkUp yet.",
  NO_GLUCOSE_UPLOADED_YET:
    "The connection exists, but no glucose data has reached the cloud yet. Keep the Libre phone online and try again.",
  NETWORK_DELAY: "We could not confirm the latest upload yet. Please retry in a few minutes.",
  UNKNOWN: "We could not verify the connection. Please restart the flow.",
};

export type VerifierInput = {
  invitedEmail: string;
  acceptedEmail?: string | null;
  acceptedAt?: Date | null;
  hasAnyData: boolean;
  lastDataAt?: Date | null;
  isOnline?: boolean;
  regionMatches?: boolean;
};

export type VerifierResult = {
  status: LibreConnectionStatus;
  errorCode: LibreDiagnosticCode | null;
  errorMessage: string | null;
  diagnostics: Record<string, unknown>;
};

export function normalizeEmail(email: string | null | undefined): string {
  return (email ?? "").trim().toLowerCase();
}

export function emailsMatch(a: string | null | undefined, b: string | null | undefined): boolean {
  return normalizeEmail(a) !== "" && normalizeEmail(a) === normalizeEmail(b);
}

export function evaluateConnectionState(input: VerifierInput): VerifierResult {
  if (!input.invitedEmail.trim()) {
    return withError("NOT_STARTED", "UNKNOWN", { reason: "missing_invited_email" });
  }

  if (!input.acceptedAt) {
    return withError("WAITING_FOR_LIBRELINKUP_ACCEPTANCE", "INVITATION_NOT_ACCEPTED", {
      acceptedAt: null,
    });
  }

  if (!emailsMatch(input.invitedEmail, input.acceptedEmail)) {
    return withError("EMAIL_MISMATCH", "EMAIL_MISMATCH", {
      invitedEmail: normalizeEmail(input.invitedEmail),
      acceptedEmail: normalizeEmail(input.acceptedEmail),
    });
  }

  if (input.isOnline === false) {
    return withError("NETWORK_OR_UPLOAD_DELAY", "NETWORK_DELAY", { reason: "offline" });
  }

  if (input.regionMatches === false) {
    return withError("NETWORK_OR_UPLOAD_DELAY", "NETWORK_DELAY", { reason: "region_mismatch_warning" });
  }

  if (!input.hasAnyData) {
    return withError("WAITING_FOR_DATA", "NO_GLUCOSE_UPLOADED_YET", {
      acceptedAt: input.acceptedAt.toISOString(),
      lastDataAt: null,
    });
  }

  return {
    status: "SYNC_ACTIVE",
    errorCode: null,
    errorMessage: null,
    diagnostics: {
      acceptedAt: input.acceptedAt.toISOString(),
      lastDataAt: input.lastDataAt?.toISOString() ?? null,
    },
  };
}

function withError(
  status: LibreConnectionStatus,
  code: LibreDiagnosticCode,
  diagnostics: Record<string, unknown>
): VerifierResult {
  return {
    status,
    errorCode: code,
    errorMessage: DIAGNOSTIC_MESSAGES[code],
    diagnostics,
  };
}
