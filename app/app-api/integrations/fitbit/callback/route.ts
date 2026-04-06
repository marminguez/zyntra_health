import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { exchangeCodeForTokens } from "@/server/integrations/fitbit/oauth";
import { encryptValue } from "@/server/security/crypto";
import { prisma } from "@/server/db/prisma";

export async function GET(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  const { searchParams } = new URL(req.url);
  const code  = searchParams.get("code");
  const state = searchParams.get("state");

  const cookieVerifier = req.cookies.get("fitbit_pkce_verifier")?.value;
  const cookieState    = req.cookies.get("fitbit_oauth_state")?.value;

  if (!code || !cookieVerifier || state !== cookieState) {
    return NextResponse.json({ error: "Invalid OAuth callback" }, { status: 400 });
  }

  const tokens = await exchangeCodeForTokens(code, cookieVerifier);

  // Find the patient for this user
  const user    = await prisma.user.findUnique({
    where:   { id: auth.userId },
    include: { patient: true },
  });

  if (!user?.patient) {
    return NextResponse.json({ error: "No patient profile found" }, { status: 404 });
  }

  const patientId = user.patient.id;

  // Encrypt and store tokens
  await prisma.integrationToken.upsert({
    where:  { patientId_provider: { patientId, provider: "fitbit" } },
    create: {
      patientId,
      provider:     "fitbit",
      accessToken:  await encryptValue(tokens.accessToken),
      refreshToken: await encryptValue(tokens.refreshToken),
      expiresAt:    tokens.expiresAt,
      scope:        tokens.scope,
    },
    update: {
      accessToken:  await encryptValue(tokens.accessToken),
      refreshToken: await encryptValue(tokens.refreshToken),
      expiresAt:    tokens.expiresAt,
      scope:        tokens.scope,
    },
  });

  // Clear PKCE cookies and redirect to dashboard
  const response = NextResponse.redirect(new URL("/dashboard?fitbit=connected", req.url));
  response.cookies.delete("fitbit_pkce_verifier");
  response.cookies.delete("fitbit_oauth_state");
  return response;
}
