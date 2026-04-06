import { prisma } from "../db/prisma";
import { SignalIngestInput } from "./schemas";
import { encryptValue, isSensitiveType } from "../security/crypto";
import { writeAudit } from "../security/audit";

const META_MAX_LEN = 5000;

function capMeta(meta: Record<string, unknown>, maxLen: number): string {
  const raw = JSON.stringify(meta);
  if (raw.length <= maxLen) return raw;
  const capped: Record<string, unknown> = {};
  let acc = 2; // "{}"
  for (const [k, v] of Object.entries(meta)) {
    const entry = JSON.stringify(k) + ":" + JSON.stringify(v) + ",";
    if (acc + entry.length > maxLen) break;
    capped[k] = v;
    acc += entry.length;
  }
  return JSON.stringify(capped);
}

/** Source → consent field mapping */
const SOURCE_CONSENT_FIELD: Record<string, "allowCGM" | "allowWearable" | "allowManual"> = {
    CGM: "allowCGM",
    WEARABLE: "allowWearable",
    MANUAL: "allowManual",
};

/**
 * Ingests a single signal with consent gating, field-level encryption, and audit logging.
 */
export async function ingestSignal(
    input: SignalIngestInput,
    userId: string,
    ip?: string | null,
    ua?: string | null
): Promise<{ id: string }> {
    // ── Consent check ────────────────────────────────────────────
    const consent = await prisma.consent.findFirst({
        where: {
            patientId: input.patientId,
            revokedAt: null,
        },
        orderBy: { grantedAt: "desc" },
    });

    if (!consent) {
        throw new ConsentError("No active consent found for patient");
    }

    const EXEMPT_SOURCES = new Set(["SYSTEM"]);

    if (!EXEMPT_SOURCES.has(input.source)) {
        const consentField = SOURCE_CONSENT_FIELD[input.source];
        if (!consentField || !consent[consentField]) {
            throw new ConsentError(`Consent does not allow source ${input.source}`);
        }
    }

    // ── Encryption ───────────────────────────────────────────────
    let value: number | null = input.value ?? null;
    let encryptedValueStr: string | null = null;

    if (value !== null && isSensitiveType(input.type)) {
        encryptedValueStr = await encryptValue(value);
        value = null; // clear plaintext
    }

    // ── Meta capping ─────────────────────────────────────────────
    let metaJson: string | undefined;
    if (input.meta) {
        metaJson = capMeta(input.meta as Record<string, unknown>, META_MAX_LEN);
    }

    // ── Persist & Audit ──────────────────────────────────────────
    const signal = await prisma.$transaction(async (tx) => {
        const s = await tx.signal.create({
            data: {
                patientId: input.patientId,
                source: input.source,
                type: input.type,
                ts: new Date(input.ts),
                value,
                encryptedValue: encryptedValueStr,
                unit: input.unit,
                metaJson,
            },
        });
        await tx.auditEvent.create({
            data: {
                userId,
                action: "SIGNAL_INGESTED",
                targetId: s.id,
                metaJson: JSON.stringify({ type: input.type, source: input.source }),
                ip: ip ?? null,
                ua: ua ?? null,
            },
        });
        return s;
    });

    return { id: signal.id };
}

/** Custom error for consent violations. */
export class ConsentError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ConsentError";
    }
}
