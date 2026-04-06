# Zyntra – Adaptive Metabolic Intelligence Layer

A secure infrastructure layer that ingests metabolic + wearable signals, stores them with consent gating, computes baseline statistics and 24/48/72-hour risk scores, and exposes API endpoints plus a minimal dashboard.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| ORM | Prisma + PostgreSQL |
| Auth | NextAuth (Credentials provider, JWT) |
| Validation | Zod |
| Encryption | libsodium-wrappers (secretbox) |
| Tests | Vitest |

## Security Features

- **RBAC** – Role-based access control (ADMIN, CLINICIAN, PATIENT, SERVICE)
- **Consent Gating** – Ingest checks latest active consent per patient
- **Field-Level Encryption** – Sensitive signal values encrypted at rest via libsodium secretbox
- **Rate Limiting** – In-memory per IP+route (60 req/min)
- **Audit Logging** – Every ingestion and risk computation is logged

## Quick Start

### 1. Start PostgreSQL

```bash
docker-compose up -d
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your secrets:

```env
DATABASE_URL="postgresql://zyntra:zyntra@localhost:5432/zyntra?schema=public"
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-random-secret-here"
ZYNTRA_FIELD_ENCRYPTION_KEY="<base64 32 bytes>"
ELEVENLABS_API_KEY="<optional-for-voice-alerts>"
ELEVENLABS_VOICE_ID="EXAVITQu4vr4xnSDxMaL"
ELEVENLABS_MODEL_ID="eleven_multilingual_v2"
```

Generate encryption key:

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

### 4. Run Migrations

```bash
npx prisma migrate dev --name init
```

### 5. Seed Database (optional)

```bash
npm run seed
```

### 6. Start Dev Server

```bash
npm run dev
```

Open [http://localhost:3000/login](http://localhost:3000/login) and sign in with any email and password `hackathon-dev-only`.

### Optional: Voice chat + spoken risk alerts

- The dashboard supports microphone input in the Zyntra chat (browser speech recognition).
- Risk alerts and chat replies can be spoken with a female voice.
- If `ELEVENLABS_API_KEY` is configured, audio is generated with ElevenLabs via `/api/zyntra/voice`.
- If ElevenLabs is not configured or fails, the app falls back to the browser `speechSynthesis` voice.

## API Endpoints

All endpoints are under `/app-api/` and require authentication. Obtain a session cookie by signing in via the web UI or using NextAuth's CSRF-protected flow.

### Ingest Signal

```bash
curl -X POST http://localhost:3000/app-api/ingest \
  -H "Content-Type: application/json" \
  -H "Cookie: next-auth.session-token=<your-session-token>" \
  -d '{
    "patientId": "<PATIENT_ID>",
    "source": "CGM",
    "ts": "2025-06-15T10:30:00.000Z",
    "type": "cgm_glucose_mgdl",
    "value": 142,
    "unit": "mg/dL"
  }'
```

### Compute Baseline

```bash
curl -X POST http://localhost:3000/app-api/baseline \
  -H "Content-Type: application/json" \
  -H "Cookie: next-auth.session-token=<your-session-token>" \
  -d '{
    "patientId": "<PATIENT_ID>",
    "windowDays": 14
  }'
```

### Compute Risk Score

```bash
curl -X POST http://localhost:3000/app-api/risk \
  -H "Content-Type: application/json" \
  -H "Cookie: next-auth.session-token=<your-session-token>" \
  -d '{
    "patientId": "<PATIENT_ID>"
  }'
```

## Running Tests

```bash
npm test
```

## Project Structure

```
zyntra/
  app/                          # Next.js App Router pages
    (auth)/login/page.tsx       # Login page
    dashboard/page.tsx          # Dashboard (risk score viewer)
    app-api/                    # Route proxies (re-export from src/)
      ingest/route.ts
      risk/route.ts
      baseline/route.ts
    api/auth/[...nextauth]/     # NextAuth handler
    layout.tsx                  # Root layout
    globals.css                 # Design system
  src/
    server/
      auth/
        auth.ts                 # NextAuth configuration
        rbac.ts                 # Role-based access control
      db/
        prisma.ts               # Singleton PrismaClient
      security/
        crypto.ts               # Field-level encryption (libsodium)
        rateLimit.ts            # In-memory rate limiter
        audit.ts                # Audit event logger
      zyntra/
        engine.ts               # Risk scoring engine
        ingest.ts               # Signal ingestion logic
        schemas.ts              # Zod validation schemas
    app-api/                    # API route implementations
      ingest/route.ts
      risk/route.ts
      baseline/route.ts
  prisma/
    schema.prisma               # Data model
  scripts/
    seed.ts                     # Database seed
  tests/
    engine.test.ts              # Engine unit tests
  docker-compose.yml
  .env.example
```

## License

Private – Harvard Project
