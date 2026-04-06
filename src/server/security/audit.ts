import { prisma } from "../db/prisma";

const META_MAX_LEN = 5000;

export interface AuditInput {
    userId?: string;
    action: string;
    targetId?: string;
    meta?: Record<string, unknown>;
    ip?: string | null;
    ua?: string | null;
}

/**
 * Writes an AuditEvent row. Stringifies and caps meta to 5000 chars.
 */
export async function writeAudit(input: AuditInput): Promise<void> {
    let metaJson: string | undefined;
    if (input.meta) {
        const raw = JSON.stringify(input.meta);
        metaJson = raw.length > META_MAX_LEN ? raw.slice(0, META_MAX_LEN) : raw;
    }

    await prisma.auditEvent.create({
        data: {
            userId: input.userId,
            action: input.action,
            targetId: input.targetId,
            metaJson,
            ip: input.ip ?? undefined,
            ua: input.ua ?? undefined,
        },
    });
}
