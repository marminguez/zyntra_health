import { describe, expect, it } from "vitest";
import { evaluateConnectionState, emailsMatch } from "@/server/integrations/freestyle/onboarding";

describe("LibreLinkUp onboarding verifier", () => {
  it("invitation sent but not accepted", () => {
    const result = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: null,
      acceptedAt: null,
      hasAnyData: false,
    });

    expect(result.status).toBe("WAITING_FOR_LIBRELINKUP_ACCEPTANCE");
    expect(result.errorCode).toBe("INVITATION_NOT_ACCEPTED");
  });

  it("accepted with different email", () => {
    const result = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: "other@example.com",
      acceptedAt: new Date("2026-01-01T00:00:00Z"),
      hasAnyData: false,
    });

    expect(result.status).toBe("EMAIL_MISMATCH");
    expect(result.errorCode).toBe("EMAIL_MISMATCH");
  });

  it("accepted but no data yet", () => {
    const result = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: "patient@example.com",
      acceptedAt: new Date("2026-01-01T00:00:00Z"),
      hasAnyData: false,
      isOnline: true,
    });

    expect(result.status).toBe("WAITING_FOR_DATA");
    expect(result.errorCode).toBe("NO_GLUCOSE_UPLOADED_YET");
  });

  it("data available and sync active", () => {
    const result = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: "patient@example.com",
      acceptedAt: new Date("2026-01-01T00:00:00Z"),
      hasAnyData: true,
      lastDataAt: new Date("2026-01-01T00:05:00Z"),
      isOnline: true,
    });

    expect(result.status).toBe("SYNC_ACTIVE");
    expect(result.errorCode).toBeNull();
  });

  it("retry after transient delay", () => {
    const delayed = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: "patient@example.com",
      acceptedAt: new Date("2026-01-01T00:00:00Z"),
      hasAnyData: false,
      isOnline: false,
    });

    const retried = evaluateConnectionState({
      invitedEmail: "patient@example.com",
      acceptedEmail: "patient@example.com",
      acceptedAt: new Date("2026-01-01T00:00:00Z"),
      hasAnyData: true,
      lastDataAt: new Date("2026-01-01T00:10:00Z"),
      isOnline: true,
    });

    expect(delayed.status).toBe("NETWORK_OR_UPLOAD_DELAY");
    expect(retried.status).toBe("SYNC_ACTIVE");
  });

  it("reset flow email matcher handles case-insensitive trim compare", () => {
    expect(emailsMatch("  User@Example.com", "user@example.com ")).toBe(true);
    expect(emailsMatch("", "user@example.com")).toBe(false);
  });
});
