import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { generatePKCE, buildAuthUrl } from "@/server/integrations/fitbit/oauth";
import crypto from "crypto";

export async function GET(req: NextRequest) {
  const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
  if (!auth.authorized) return auth.response;

  try {
    const { verifier, challenge } = generatePKCE();
    const state = crypto.randomBytes(16).toString("hex");

    // Store verifier + state in a short-lived cookie (5 min)
    const url = buildAuthUrl(state, challenge);
    const response = NextResponse.redirect(url);
    response.cookies.set("fitbit_pkce_verifier", verifier, { httpOnly: true, maxAge: 300, path: "/" });
    response.cookies.set("fitbit_oauth_state", state, { httpOnly: true, maxAge: 300, path: "/" });

    return response;
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to start Fitbit OAuth";
    return NextResponse.redirect(new URL(`/dashboard?fitbit=error&message=${encodeURIComponent(message)}`, req.url));
  }
}
