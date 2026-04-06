import crypto from "crypto";
import { FITBIT_CONFIG, hasFitbitCredentials } from "./config";

/** Generates PKCE code_verifier + code_challenge */
export function generatePKCE(): { verifier: string; challenge: string } {
  const verifier = crypto.randomBytes(32).toString("base64url");
  const challenge = crypto
    .createHash("sha256")
    .update(verifier)
    .digest("base64url");
  return { verifier, challenge };
}

/** Builds the Fitbit OAuth2 authorisation URL */
export function buildAuthUrl(state: string, codeChallenge: string): string {
  if (!hasFitbitCredentials()) {
    throw new Error("Fitbit OAuth is not configured (missing FITBIT_CLIENT_ID/SECRET/REDIRECT_URI)");
  }

  const params = new URLSearchParams({
    client_id:             FITBIT_CONFIG.clientId,
    response_type:         "code",
    scope:                 FITBIT_CONFIG.scopes.join(" "),
    redirect_uri:          FITBIT_CONFIG.redirectUri,
    state,
    code_challenge:        codeChallenge,
    code_challenge_method: "S256",
  });
  return `${FITBIT_CONFIG.authUrl}?${params.toString()}`;
}

/** Exchanges authorisation code for access + refresh tokens */
export async function exchangeCodeForTokens(
  code: string,
  codeVerifier: string
): Promise<{ accessToken: string; refreshToken: string; expiresAt: Date; scope: string }> {
  if (!hasFitbitCredentials() || FITBIT_CONFIG.clientId === "fitbit_client_id_placeholder") {
    // Return mock tokens for local testing
    return {
      accessToken: "mock_access_token",
      refreshToken: "mock_refresh_token",
      expiresAt: new Date(Date.now() + 8 * 3600 * 1000), // 8 hours
      scope: "activity heartrate sleep nutrition profile weight",
    };
  }

  const credentials = Buffer.from(
    `${FITBIT_CONFIG.clientId}:${FITBIT_CONFIG.clientSecret}`
  ).toString("base64");

  const res = await fetch(FITBIT_CONFIG.tokenUrl, {
    method: "POST",
    headers: {
      "Authorization": `Basic ${credentials}`,
      "Content-Type":  "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      client_id:     FITBIT_CONFIG.clientId,
      grant_type:    "authorization_code",
      redirect_uri:  FITBIT_CONFIG.redirectUri,
      code,
      code_verifier: codeVerifier,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Fitbit token exchange failed: ${err}`);
  }

  const data = await res.json();
  const expiresAt = new Date(Date.now() + data.expires_in * 1000);

  return {
    accessToken:  data.access_token,
    refreshToken: data.refresh_token,
    expiresAt,
    scope:        data.scope,
  };
}

/** Refreshes an expired access token */
export async function refreshAccessToken(
  refreshToken: string
): Promise<{ accessToken: string; refreshToken: string; expiresAt: Date }> {
  if (!hasFitbitCredentials()) {
    return {
      accessToken: "mock_access_token",
      refreshToken: "mock_refresh_token",
      expiresAt: new Date(Date.now() + 8 * 3600 * 1000),
    };
  }
  const credentials = Buffer.from(
    `${FITBIT_CONFIG.clientId}:${FITBIT_CONFIG.clientSecret}`
  ).toString("base64");

  const res = await fetch(FITBIT_CONFIG.tokenUrl, {
    method: "POST",
    headers: {
      "Authorization": `Basic ${credentials}`,
      "Content-Type":  "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      grant_type:    "refresh_token",
      refresh_token: refreshToken,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Fitbit token refresh failed: ${err}`);
  }

  const data = await res.json();
  return {
    accessToken:  data.access_token,
    refreshToken: data.refresh_token,
    expiresAt:    new Date(Date.now() + data.expires_in * 1000),
  };
}
