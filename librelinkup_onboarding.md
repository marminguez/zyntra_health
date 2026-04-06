# LibreLinkUp onboarding flow

## Summary
Zyntra supports two practical paths:
1. Follower/share onboarding (Connected Apps invite + LibreLinkUp acceptance).
2. Direct sync checks when the user logs in to Zyntra with Libre credentials.

## Follower/share state machine
- `NOT_STARTED`
- `INVITE_SENT`
- `WAITING_FOR_LIBRELINKUP_ACCEPTANCE`
- `SHARE_ACCEPTED_NO_DATA_YET`
- `WAITING_FOR_DATA`
- `SYNC_ACTIVE`
- `SYNC_ERROR`
- `EMAIL_MISMATCH`
- `NETWORK_OR_UPLOAD_DELAY`

## Current behavior
- If a freestyle credential token is available, `Check connection` attempts direct LibreLinkUp sync and ingests readings.
- If no credential token is available, Zyntra falls back to the follower/share verifier flow.

## How this maps to unofficial LibreLinkUp APIs
- Zyntra already uses an unofficial LibreLinkUp client (`@diakem/libre-link-up-api-client`) to authenticate and read glucose graph data.
- The direct-sync path is compatible with the technical approach based on LibreView/LibreLinkUp reverse-engineered endpoints.
- Because this path depends on non-public APIs, users may still see temporary auth/rate-limit errors and should be prompted to retry.

## Practical setup for a patient in Zyntra
1. In LibreLinkUp, make sure cloud sharing is enabled and data is actively uploading from the Libre app.
2. In Zyntra Connected Apps, connect FreeStyle with the same LibreLinkUp credentials.
3. Run `Check connection` and confirm status moves to `SYNC_ACTIVE`.
4. If status remains `WAITING_FOR_DATA`, wait for a fresh sensor upload and retry.

## Known limitations
- Direct login/sync depends on LibreLinkUp API behavior and can fail on auth/rate limits.
- Follower/share mode may remain in waiting states until cloud glucose uploads become available.
