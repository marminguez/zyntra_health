export const FITBIT_CONFIG = {
  clientId:     process.env.FITBIT_CLIENT_ID ?? "",
  clientSecret: process.env.FITBIT_CLIENT_SECRET ?? "",
  redirectUri:  process.env.FITBIT_REDIRECT_URI ?? "",
  authUrl:      "https://www.fitbit.com/oauth2/authorize",
  tokenUrl:     "https://api.fitbit.com/oauth2/token",
  scopes:       ["activity", "heartrate", "sleep", "oxygen_saturation", "respiratory_rate", "profile"],
} as const;

export const FITBIT_API = "https://api.fitbit.com/1/user/-";

export function hasFitbitCredentials(): boolean {
  return Boolean(FITBIT_CONFIG.clientId && FITBIT_CONFIG.clientSecret && FITBIT_CONFIG.redirectUri);
}
